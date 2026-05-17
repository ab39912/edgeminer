"""
Uncertainty scoring for EdgeMiner active learning.

Given YOLOv8 detections on a dataset, compute per-scene uncertainty scores
using multiple strategies. Scenes with high uncertainty are candidates for
human review and labeling — they are "the data that breaks the model."

Implemented strategies:
    1. mean_confidence: average detection confidence (lower = more uncertain)
    2. min_confidence: weakest detection in the scene (lower = more uncertain)
    3. low_conf_count: number of detections below a threshold
    4. entropy: Shannon entropy over class distribution (higher = more uncertain)
    5. no_detection: scenes where the detector found nothing
    6. composite: weighted combination of the above

A single "uncertainty score" per scene, normalized to [0, 1], makes ranking easy.
"""

from typing import Dict, List

import numpy as np
import torch


def compute_scene_uncertainty(detection: dict) -> dict:
    """
    Compute multiple uncertainty signals for a single scene's detections.

    Args:
        detection: dict with keys 'confidences', 'classes', 'num_detections', etc.

    Returns:
        dict with individual uncertainty scores (higher = more uncertain everywhere)
    """
    confidences = detection["confidences"]
    num_detections = detection["num_detections"]

    if num_detections == 0:
        # No detections is its own uncertainty signal: detector missed everything
        return {
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "uncertainty_mean": 1.0,        # 1 - mean_conf, max because no detections
            "uncertainty_min": 1.0,
            "low_conf_count": 0,
            "low_conf_ratio": 0.0,
            "class_entropy": 0.0,
            "is_no_detection": 1.0,
            "num_detections": 0,
        }

    confs = confidences.numpy() if isinstance(confidences, torch.Tensor) else np.asarray(confidences)
    classes = detection["classes"]
    if isinstance(classes, torch.Tensor):
        classes = classes.numpy()

    mean_conf = float(confs.mean())
    min_conf = float(confs.min())

    # Detections with confidence < 0.5 are considered "weak"
    low_conf_mask = confs < 0.5
    low_conf_count = int(low_conf_mask.sum())
    low_conf_ratio = float(low_conf_count / len(confs))

    # Class entropy: scenes with rare class mixes are more uncertain
    unique, counts = np.unique(classes, return_counts=True)
    probs = counts / counts.sum()
    class_entropy = float(-(probs * np.log(probs + 1e-10)).sum())

    return {
        "mean_confidence": mean_conf,
        "min_confidence": min_conf,
        "uncertainty_mean": 1.0 - mean_conf,
        "uncertainty_min": 1.0 - min_conf,
        "low_conf_count": low_conf_count,
        "low_conf_ratio": low_conf_ratio,
        "class_entropy": class_entropy,
        "is_no_detection": 0.0,
        "num_detections": num_detections,
    }


def compute_dataset_uncertainty(
    detections: Dict[str, dict],
    composite_weights: dict = None,
) -> Dict[str, dict]:
    """
    Compute uncertainty for every scene in the dataset.

    Also computes a normalized 'composite_score' in [0, 1] per scene, defined as
    a weighted average of the individual uncertainty signals.

    Args:
        detections: dict of {sample_token: detection_summary}
        composite_weights: dict mapping signal name -> weight. Defaults to equal weights.

    Returns:
        dict of {sample_token: full_uncertainty_dict} including composite_score
    """
    if composite_weights is None:
        composite_weights = {
            "uncertainty_mean": 0.30,
            "uncertainty_min": 0.20,
            "low_conf_ratio": 0.20,
            "class_entropy": 0.10,
            "is_no_detection": 0.20,
        }

    # First pass: compute raw signals
    scores = {}
    for token, det in detections.items():
        scores[token] = compute_scene_uncertainty(det)

    # Normalize each signal to [0, 1] across the dataset (min-max)
    signal_names = list(composite_weights.keys())
    for signal in signal_names:
        values = np.array([scores[t][signal] for t in scores])
        v_min, v_max = float(values.min()), float(values.max())
        if v_max > v_min:
            for t in scores:
                scores[t][f"{signal}_norm"] = (scores[t][signal] - v_min) / (v_max - v_min)
        else:
            for t in scores:
                scores[t][f"{signal}_norm"] = 0.0

    # Composite score: weighted average of normalized signals
    for t in scores:
        composite = sum(
            scores[t][f"{signal}_norm"] * w
            for signal, w in composite_weights.items()
        )
        scores[t]["composite_score"] = float(composite)

    return scores


def rank_by_uncertainty(
    uncertainty_scores: Dict[str, dict],
    top_k: int = 20,
    score_key: str = "composite_score",
) -> List[tuple]:
    """
    Return the top-K most uncertain scenes.

    Returns: list of (sample_token, score) sorted descending by score.
    """
    ranked = sorted(
        uncertainty_scores.items(),
        key=lambda x: x[1][score_key],
        reverse=True,
    )
    return [(token, scores[score_key]) for token, scores in ranked[:top_k]]


def summarize_uncertainty(uncertainty_scores: Dict[str, dict]):
    """Print a quick summary of the uncertainty distribution."""
    composite = np.array([s["composite_score"] for s in uncertainty_scores.values()])
    no_det_count = sum(1 for s in uncertainty_scores.values() if s["is_no_detection"] > 0.5)

    print(f"Uncertainty summary over {len(uncertainty_scores)} scenes:")
    print(f"  Composite score    min={composite.min():.3f}, "
          f"median={np.median(composite):.3f}, "
          f"mean={composite.mean():.3f}, "
          f"max={composite.max():.3f}")
    print(f"  Scenes with zero detections: {no_det_count}")
    print(f"  Top 10% threshold: {np.percentile(composite, 90):.3f}")
