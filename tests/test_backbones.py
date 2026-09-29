"""Small, weight-free checks of the frozen-model adapter's public boundaries."""

import importlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch


class BackboneBoundaryTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("hec_cd.backbones")
        except ModuleNotFoundError as exc:
            self.fail(f"Frozen backbone adapter is not implemented: {exc}")

    def test_rejects_non_rgb_and_wrong_size_without_loading_weights(self):
        valid = Image.new("RGB", (256, 256))
        self.module.validate_pair(valid, valid)
        for invalid in (Image.new("L", (256, 256)), Image.new("RGBA", (256, 256)),
                        Image.new("RGB", (512, 256))):
            with self.subTest(mode=invalid.mode, size=invalid.size):
                with self.assertRaises(ValueError):
                    self.module.validate_pair(valid, invalid)

    def test_rgb_tensor_keeps_channel_order_and_uint8_scale(self):
        pixels = np.zeros((256, 256, 3), dtype=np.uint8)
        pixels[0, 0] = (0, 128, 255)
        tensor = self.module._image_tensor(Image.fromarray(pixels))
        self.assertEqual(tuple(tensor.shape), (1, 3, 256, 256))
        # The research DataLoader stacks into contiguous NCHW; a channels-last
        # view can choose different convolution kernels despite identical RGB.
        self.assertTrue(tensor.is_contiguous())
        torch.testing.assert_close(tensor[0, :, 0, 0], torch.tensor([0., 128 / 255, 1.]))

    def test_satlas_residual_truncates_before_returning_normalized_score(self):
        first = torch.tensor([[[[0.001, 0.501, 0.9999]]]])
        second = torch.zeros_like(first)
        actual = self.module._residual_map(first, second)
        np.testing.assert_array_equal(actual, np.array([[0., 127 / 255, 254 / 255]], np.float32))

    def test_prompt_semantics_round_and_presence_half_precision_precede_max(self):
        # Rounding 127.5 to 128 and rounding 0.5001 to float16 0.5 are
        # both necessary to reproduce the on-disk research cache boundary.
        first = self.module._semantic_u8(torch.full((1, 1, 2, 2), 0.5))
        second = self.module._semantic_u8(torch.full((1, 1, 2, 2), 0.9))
        response = self.module._positive_response([first, second], [0.5001, 0.25])
        np.testing.assert_array_equal(response, np.full((2, 2), 64 / 255, np.float32))

    def test_constructor_requires_explicit_existing_local_checkpoints(self):
        missing = Path(__file__).parent / "does-not-exist.pth"
        with self.assertRaises(FileNotFoundError):
            self.module.FrozenBackbones(missing, missing.parent, missing, missing, device="cpu")

    def test_feature_pair_preserves_inference_device_and_detaches(self):
        # The meta device checks device retention without allocating GPU memory.
        features = torch.empty((2, 3, 16, 16), device="meta", requires_grad=True)
        self.assertTrue(hasattr(self.module, "_feature_pair"))
        first, second = self.module._feature_pair(features)
        for values in (first, second):
            self.assertEqual(len(values), 1)
            self.assertEqual(tuple(values[0].shape), (3, 16, 16))
            self.assertEqual(values[0].device.type, "meta")
            self.assertFalse(values[0].requires_grad)

    def test_sam_import_failure_restores_global_tf32_flags(self):
        adapter = object.__new__(self.module.FrozenBackbones)
        adapter.device = torch.device("cpu")
        saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

        def failed_external_loader():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = False
            raise RuntimeError("external SAM3 initialization failed")

        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
            with patch.object(adapter, "_load_sam", side_effect=failed_external_loader):
                with self.assertRaisesRegex(RuntimeError, "external SAM3"):
                    adapter.infer_temporal(Image.new("RGB", (256, 256)), Image.new("RGB", (256, 256)))
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved

    def test_sam_moves_model_to_indexed_device_before_encoding_text(self):
        class ExternalModel:
            def __init__(self):
                self.location = "cpu"
                self.backbone = SimpleNamespace(forward_text=self.forward_text)

            def to(self, device):
                self.location = str(device)
                return self

            def requires_grad_(self, enabled):
                return self

            def eval(self):
                return self

            def forward_text(self, prompts, device):
                if self.location != device:
                    raise RuntimeError("model parameters and text tokens are on different devices")
                return {"language_features": torch.zeros(1)}

        adapter = object.__new__(self.module.FrozenBackbones)
        adapter.device = torch.device("cuda:0")
        adapter.sam3_root = Path(__file__).parent.resolve()
        adapter.sam3_bpe = adapter.sam3_root / "vocab.gz"
        adapter.sam3_checkpoint = adapter.sam3_root / "sam3.pt"
        adapter._sam_model = None
        package = SimpleNamespace(__file__=str(adapter.sam3_root / "sam3.py"),
                                  build_sam3_image_model=lambda **kwargs: ExternalModel())
        processor = SimpleNamespace(Sam3Processor=lambda *args, **kwargs: object())
        with patch.object(self.module.importlib, "import_module", side_effect=[package, processor]):
            try:
                adapter._load_sam()
            except RuntimeError as exc:
                self.fail(str(exc))
        # A text prompt can now be encoded on the requested indexed device.
        result = adapter._sam_model.backbone.forward_text(["building"], device="cuda:0")
        self.assertIn("language_features", result)

    def test_temporal_inference_uses_each_image_and_its_text_prompt_context(self):
        class Processor:
            find_stage = SimpleNamespace(img_ids=torch.tensor([0]))

            def set_image_batch(self, images):
                raise AssertionError("Historical evidence requires separate image-backbone passes")

            def set_image(self, image):
                return {"response": 0.5 if image.getpixel((0, 0))[0] == 0 else 0.9}

            def reset_all_prompts(self, state):
                pass

            def set_text_prompt(self, prompt, state):
                return {**state, "semantic_mask_logits": torch.full((1, 1, 256, 256), state["response"]),
                        "presence_score": torch.tensor(0.5001),
                        "masks": torch.zeros((0, 1, 256, 256), dtype=torch.bool),
                        "scores": torch.zeros(0)}

        class Encoder(torch.nn.Module):
            def forward(self, images):
                return [torch.ones((2, 3, 16, 16))]

        adapter = object.__new__(self.module.FrozenBackbones)
        adapter.device = torch.device("cpu")
        adapter._sam_model = torch.nn.Identity()
        with patch.object(adapter, "_load_sam", return_value=Processor()), \
             patch.object(adapter, "_load_dino", return_value=Encoder()):
            result = adapter.infer_temporal(Image.new("RGB", (256, 256)),
                                            Image.new("RGB", (256, 256), color=(1, 0, 0)))
        np.testing.assert_array_equal(result["p1"], np.full((256, 256), 64 / 255, np.float32))
        np.testing.assert_array_equal(result["p2"], np.full((256, 256), 115 / 255, np.float32))
        self.assertEqual(result["instances1"].shape, (0, 256, 256))


if __name__ == "__main__":
    unittest.main()
