"""Frozen Satlas, SegEarth-OV-3 SAM3, and DINOv3 inference adapters.

Checkpoints and third-party source trees are supplied by the caller. Models
are loaded lazily and moved back to CPU between stages to limit GPU memory.
The quantization boundaries intentionally reproduce the research pipeline.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn


PROMPTS = ("building", "rooftop", "house", "urban structure")
SAM3_SOURCE_COMMIT = "87e5cc18c8bc9c0b4713b975587281ac1c97c79a"
SATELLITE_MEAN = (0.430, 0.411, 0.296)
SATELLITE_STD = (0.213, 0.156, 0.143)


def validate_pair(first: Image.Image, second: Image.Image) -> None:
    """Require registered 256-pixel RGB tiles without resizing or conversion."""
    for name, image in (("first", first), ("second", second)):
        if not isinstance(image, Image.Image):
            raise TypeError(f"{name} must be a PIL image")
        if image.mode != "RGB" or image.size != (256, 256):
            raise ValueError(f"{name} must be a 256x256 RGB image; got {image.mode} {image.size}")


def _image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def _residual_map(first: torch.Tensor, second: torch.Tensor) -> np.ndarray:
    # The original Satlas cache uses truncation, not round-to-nearest.
    residual_u8 = (first - second).abs()[0, 0].clamp(0, 1).mul(255).byte().cpu().numpy()
    return residual_u8.astype(np.float32) / 255.0


def _semantic_u8(value: torch.Tensor) -> np.ndarray:
    # SegEarth's misleadingly named semantic_mask_logits is already sigmoid.
    array = value.detach().float().squeeze().clamp(0, 1).cpu().numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def _positive_response(semantics: list[np.ndarray], presence: list[float]) -> np.ndarray:
    probabilities = np.stack(semantics).astype(np.float32) / 255.0
    scores = np.asarray(presence, dtype=np.float16).astype(np.float32)
    return (probabilities * scores[:, None, None]).max(axis=0)


def _feature_pair(feature: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    # Pooling stays on the inference device, as in the original DINO script.
    return [feature[0].detach()], [feature[1].detach()]


@contextmanager
def _sam_precision(device: torch.device):
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        # The pinned SAM3 builder enables both flags on Ampere. Repeat this for
        # cached models as well, but keep its global side effect inside SAM.
        if device.type == "cuda" and torch.cuda.get_device_properties(device).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _component_state(state: dict, prefix: str) -> dict:
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


class _SatlasBuildings(nn.Module):
    """Official SI building instance head with the frozen inference settings."""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        from satlaspretrain_models.models.backbones import SwinBackbone
        from satlaspretrain_models.models.fpn import FPN
        from satlaspretrain_models.models.heads import FRCNNHead

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.backbone = SwinBackbone(3, arch="swinb")
        self.backbone.load_state_dict(_component_state(state, "backbone."), strict=True)
        self.fpn = FPN(self.backbone.out_channels)
        self.fpn.load_state_dict(_component_state(state, "intermediates.0."), strict=True)
        self.head = FRCNNHead("instance", self.fpn.out_channels, num_categories=2)
        self.head.load_state_dict(_component_state(state, "heads.14."), strict=True)
        self.head.rpn._pre_nms_top_n["testing"] = 600
        self.head.rpn._post_nms_top_n["testing"] = 200
        self.head.roi_heads.detections_per_img = 50
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
        self.requires_grad_(False).eval()

    def occupancy(self, normalized: torch.Tensor) -> torch.Tensor:
        from torchvision.models.detection.roi_heads import paste_masks_in_image

        # Keep the formal cache's ImageNet-normalize -> denormalize sequence.
        raw = (normalized * self.std + self.mean).clamp(0, 1)
        detections, _ = self.head(raw, self.fpn(self.backbone(raw)))
        size = tuple(raw.shape[-2:])
        outputs = []
        for detection in detections:
            selected = (detection["labels"] == 1) & (detection["scores"] >= 0.20)
            if not selected.any():
                outputs.append(raw.new_zeros((1, *size)))
                continue
            masks = paste_masks_in_image(detection["masks"][selected], detection["boxes"][selected], size)
            confidence = detection["scores"][selected].view(-1, 1, 1, 1)
            outputs.append((masks * confidence).amax(dim=0))
        return torch.stack(outputs, dim=0)


class FrozenBackbones:
    """Generate the two-time evidence used by HEC-CD from a raw RGB pair.

    ``infer_pair`` returns NumPy maps/instance masks and one device feature tensor
    per timestamp. ``infer_satlas`` and ``infer_temporal`` can also be called
    separately; a caller may skip temporal inference for zero-Rs tiles when
    only the final fused score is needed, but then Ro itself is not measured.
    """

    def __init__(
        self,
        satlas_checkpoint: Path,
        sam3_root: Path,
        sam3_checkpoint: Path,
        dinov3_checkpoint: Path,
        device: str = "cuda",
        *,
        satlas_root: Path | None = None,
    ) -> None:
        self.satlas_checkpoint = Path(satlas_checkpoint).resolve()
        self.sam3_root = Path(sam3_root).resolve()
        self.sam3_checkpoint = Path(sam3_checkpoint).resolve()
        self.dinov3_checkpoint = Path(dinov3_checkpoint).resolve()
        self.sam3_bpe = self.sam3_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        for path in (self.satlas_checkpoint, self.sam3_checkpoint, self.dinov3_checkpoint, self.sam3_bpe):
            if not path.is_file():
                raise FileNotFoundError(f"Required local model file is missing: {path}")
        if satlas_root is not None:
            satlas_root = Path(satlas_root).resolve()
            if not (satlas_root / "satlaspretrain_models").is_dir():
                raise FileNotFoundError(f"satlas_root must contain satlaspretrain_models: {satlas_root}")
            if str(satlas_root) not in sys.path:
                sys.path.insert(0, str(satlas_root))
        self.device = torch.device(device)
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("Only CUDA and CPU devices are supported")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a CUDA-enabled PyTorch environment")
        self._satlas = None
        self._sam_model = None
        self._sam_processor = None
        self._dino = None

    def _load_satlas(self) -> _SatlasBuildings:
        if self._satlas is None:
            self._satlas = _SatlasBuildings(self.satlas_checkpoint)
        return self._satlas.to(self.device)

    def _load_sam(self):
        if self._sam_model is None:
            if str(self.sam3_root) not in sys.path:
                sys.path.insert(0, str(self.sam3_root))
            sam3 = importlib.import_module("sam3")
            if not Path(sam3.__file__).resolve().is_relative_to(self.sam3_root):
                raise RuntimeError("A different sam3 package is already imported; use the specified SegEarth-OV-3 checkout in a fresh process")
            processor_module = importlib.import_module("sam3.model.sam3_image_processor")
            self._sam_model = sam3.build_sam3_image_model(
                bpe_path=str(self.sam3_bpe), checkpoint_path=str(self.sam3_checkpoint),
                load_from_HF=False, device=str(self.device), eval_mode=True,
            ).to(self.device).requires_grad_(False).eval()
            self._sam_processor = processor_module.Sam3Processor(
                self._sam_model, confidence_threshold=0.20, device=str(self.device),
            )
        self._sam_model.to(self.device)
        return self._sam_processor

    def _load_dino(self):
        if self._dino is None:
            import timm

            self._dino = timm.create_model(
                "vit_large_patch16_dinov3.sat493m", features_only=True,
                out_indices=(23,), pretrained=True,
                pretrained_cfg_overlay={"file": str(self.dinov3_checkpoint)},
                dynamic_img_size=True,
            ).requires_grad_(False).eval()
        return self._dino.to(self.device)

    @torch.inference_mode()
    def infer_satlas(self, first: Image.Image, second: Image.Image) -> np.ndarray:
        """Return Rs after the original uint8 truncation and division by 255."""
        validate_pair(first, second)
        model = self._load_satlas()
        try:
            first_raw = _image_tensor(first).to(self.device)
            second_raw = _image_tensor(second).to(self.device)
            first_occupancy = model.occupancy((first_raw - model.mean) / model.std)
            second_occupancy = model.occupancy((second_raw - model.mean) / model.std)
            return _residual_map(first_occupancy, second_occupancy)
        finally:
            model.cpu()

    def _sam_view(self, processor, state: dict) -> tuple[np.ndarray, np.ndarray]:
        semantics, presence, masks, scores = [], [], [], []
        for prompt in PROMPTS:
            processor.reset_all_prompts(state)
            # Encode text inside this image's BF16 context to preserve precision.
            state = processor.set_text_prompt(prompt, state)
            if "semantic_mask_logits" not in state or "presence_score" not in state:
                raise RuntimeError("SAM3 processor lacks the SegEarth-OV-3 semantic/presence API; use the documented source revision")
            semantics.append(_semantic_u8(state["semantic_mask_logits"]))
            presence.append(float(state["presence_score"].detach().float().cpu()))
            prompt_masks = state["masks"].detach().squeeze(1).cpu().numpy().astype(np.uint8)
            prompt_scores = state["scores"].detach().float().cpu().numpy()
            if len(prompt_masks):
                masks.append(prompt_masks)
                scores.append(prompt_scores)
        if masks:
            all_masks, all_scores = np.concatenate(masks), np.concatenate(scores)
            instances = all_masks[np.argsort(all_scores)[::-1][:48]].astype(bool)
        else:
            instances = np.zeros((0, 256, 256), dtype=bool)
        return _positive_response(semantics, presence), instances

    @torch.inference_mode()
    def infer_temporal(self, first: Image.Image, second: Image.Image) -> dict:
        """Return SAM responses/instances and DINO last-layer C,H,W features."""
        validate_pair(first, second)
        views = []
        with _sam_precision(self.device):
            processor = self._load_sam()
            try:
                for image in (first, second):
                    # Process each image separately to preserve BF16 arithmetic.
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                        state = processor.set_image(image)
                        views.append(self._sam_view(processor, state))
            finally:
                processor.find_stage.img_ids = torch.tensor([0], device=self.device, dtype=torch.long)
                self._sam_model.cpu()

        encoder = self._load_dino()
        try:
            images = torch.cat((_image_tensor(first), _image_tensor(second)), dim=0).to(self.device)
            mean = torch.tensor(SATELLITE_MEAN, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(SATELLITE_STD, device=self.device).view(1, 3, 1, 1)
            normalized = (images - mean) / std
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                feature = encoder(normalized)[-1]
            features1, features2 = _feature_pair(feature)
            return {"p1": views[0][0], "p2": views[1][0],
                    "instances1": views[0][1], "instances2": views[1][1],
                    "features1": features1, "features2": features2}
        finally:
            encoder.cpu()

    @torch.inference_mode()
    def infer_pair(self, first: Image.Image, second: Image.Image) -> dict:
        """Run all three backbones and return unthresholded pair evidence."""
        validate_pair(first, second)
        residual = self.infer_satlas(first, second)
        return {"rs": residual, **self.infer_temporal(first, second)}
