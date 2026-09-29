"""SAM3 candidate masks and DINOv3 object change evidence."""

import numpy as np
from scipy.ndimage import label
import torch
import torch.nn.functional as F


def _components(mask: np.ndarray) -> list[np.ndarray]:
    labels, count = label(mask)
    if not count:
        return []
    areas = np.bincount(labels.ravel())
    return [labels == index for index in range(1, count + 1) if areas[index] >= 20]


def proposals(instances_t1: np.ndarray, instances_t2: np.ndarray,
              probability_t1: np.ndarray, probability_t2: np.ndarray) -> list[np.ndarray]:
    candidates = [
        mask.astype(bool)
        for mask in np.concatenate((instances_t1, instances_t2), axis=0)
        if int(mask.sum()) >= 20
    ]
    candidates.extend(_components(probability_t1 >= 0.30))
    candidates.extend(_components(probability_t2 >= 0.30))
    retained = []
    for candidate in sorted(candidates, key=lambda value: int(value.sum()), reverse=True):
        duplicate = False
        for existing in retained:
            intersection = int((candidate & existing).sum())
            union = int((candidate | existing).sum())
            if intersection / max(union, 1) >= 0.85:
                duplicate = True
                break
        if not duplicate:
            retained.append(candidate)
    return retained[:96]


@torch.inference_mode()
def mask_distances(masks: list[np.ndarray], first_features: list[torch.Tensor],
                   second_features: list[torch.Tensor]) -> np.ndarray:
    if not masks:
        return np.zeros((0,), dtype=np.float32)
    mask_tensor = torch.from_numpy(np.stack(masks).astype(np.float32)).to(first_features[0].device)
    layer_distances = []
    for first, second in zip(first_features, second_features, strict=True):
        weights = F.interpolate(mask_tensor[:, None], size=first.shape[-2:], mode="area")[:, 0]
        normalizer = weights.sum(dim=(-2, -1), keepdim=False).clamp_min(1e-3)
        pooled_first = torch.einsum("nhw,chw->nc", weights, first.float()) / normalizer[:, None]
        pooled_second = torch.einsum("nhw,chw->nc", weights, second.float()) / normalizer[:, None]
        layer_distances.append(
            (1.0 - F.cosine_similarity(pooled_first, pooled_second, dim=1)).clamp(0.0, 1.0)
        )
    return torch.stack(layer_distances).mean(dim=0).cpu().numpy()


def object_evidence(instances_t1: np.ndarray, instances_t2: np.ndarray,
                    probability_t1: np.ndarray, probability_t2: np.ndarray,
                    first_features: list[torch.Tensor], second_features: list[torch.Tensor]) -> np.ndarray:
    """Render sqrt(normalized object distance * semantic change), taking mask maxima."""
    masks = proposals(instances_t1, instances_t2, probability_t1, probability_t2)
    distances = mask_distances(masks, first_features, second_features)
    result = np.zeros_like(probability_t1, dtype=np.float32)
    for mask, distance in zip(masks, distances, strict=True):
        area = max(int(mask.sum()), 1)
        first_mean = float(probability_t1[mask].sum()) / area
        second_mean = float(probability_t2[mask].sum()) / area
        delta = abs(first_mean - second_mean)
        value = np.sqrt(min(float(distance) / 0.20, 1.0) * delta)
        np.maximum(result, mask * value, out=result)
    return result
