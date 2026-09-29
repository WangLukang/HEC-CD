"""Frozen HEC-CD fusion, target calibration, and spatial decoding."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_opening

from .ght import regularized_spec


@dataclass(frozen=True)
class ThresholdResult:
    selected_u8: int
    selected_score_01: float
    nonzero_count: int
    ght_split_u8: float | None
    parameters: dict[str, float]


def quantize(score: np.ndarray) -> np.ndarray:
    """Use fixed uint8 units; floating inputs represent scores in [0, 1]."""
    value = np.asarray(score)
    if np.issubdtype(value.dtype, np.floating):
        return np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(value, 0, 255).astype(np.uint8)


def harmonic_consensus(satlas_residual: np.ndarray, object_evidence: np.ndarray) -> np.ndarray:
    satlas = np.asarray(satlas_residual, dtype=np.float32)
    objects = np.asarray(object_evidence, dtype=np.float32)
    if satlas.shape != objects.shape:
        raise ValueError("satlas_residual and object_evidence must share a shape")
    satlas = np.clip(satlas, 0.0, 1.0)
    objects = np.clip(objects, 0.0, 1.0)
    denominator = satlas + objects
    return np.where(
        denominator > 0.0,
        2.0 * satlas * objects / np.maximum(denominator, 1e-6),
        0.0,
    ).astype(np.float32, copy=False)


def select_unlabeled_histogram(histogram: np.ndarray) -> ThresholdResult:
    """Select fixed-rho GHT from the complete unlabeled target histogram."""
    spec = regularized_spec(histogram, rho=10.0)
    cutoff = spec["low_u8"]
    return ThresholdResult(
        selected_u8=cutoff,
        selected_score_01=cutoff / 255.0,
        nonzero_count=int(spec["ght_parameters"]["nonzero_count"]),
        ght_split_u8=spec["ght_split_u8"],
        parameters=spec["ght_parameters"],
    )


def postprocess(score: np.ndarray, threshold: int) -> np.ndarray:
    """Apply inclusive cutoff followed by one 3x3 cross opening."""
    cross = np.array([[False, True, False], [True, True, True], [False, True, False]])
    return binary_opening(quantize(score) >= threshold, structure=cross, iterations=1)


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "oa": (tp + tn) / max(tp + fp + fn + tn, 1),
    }
