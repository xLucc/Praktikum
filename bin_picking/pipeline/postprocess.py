from __future__ import annotations

import numpy as np
from ultralytics.engine.results import Results


def results_to_masks(results: list[Results]) -> list[dict]:
    '''Flattens ultralytics Results into per-detection dicts: confidence, class_id, class_name, mask.'''
    detections = []

    for result in results:
        if result.masks is None:
            continue

        boxes = result.boxes

        for i, mask in enumerate(result.masks.data):
            detections.append({
                'confidence': float(boxes.conf[i]),
                'class_id': int(boxes.cls[i]),
                'class_name': result.names[int(boxes.cls[i])],
                'mask': (mask.cpu().numpy() > 0.5),
            })

    return detections


def filter_by_confidence(detections: list[dict], conf: float) -> list[dict]:
    """Keep only detections whose confidence score is at or above ``conf``."""
    return [d for d in detections if d['confidence'] >= conf]


def filter_by_percentile(detections: list[dict], alpha: float, min_detections: int = 5) -> list[dict]:
    """
    Remove low-confidence detections below the ``alpha`` quantile.

    When fewer than ``min_detections`` are present the filter is skipped entirely
    (too few samples to compute a meaningful quantile). If strict filtering would
    leave fewer than ``min_detections``, the threshold is relaxed just enough to
    keep exactly ``min_detections``.

    Args:
        detections:     List of detection dicts (must contain ``'confidence'``).
        alpha:          Quantile in [0, 1]. Detections below this quantile are dropped.
        min_detections: Minimum number of detections to retain regardless of the quantile.
    """
    if len(detections) <= min_detections:
        return detections

    confidences = [d['confidence'] for d in detections]
    threshold = np.quantile(confidences, alpha)
    kept = [d for d in detections if d['confidence'] >= threshold]

    if len(kept) < min_detections:
        relaxed = 1.0 - (min_detections / len(detections))
        threshold = np.quantile(confidences, relaxed)
        kept = [d for d in detections if d['confidence'] >= threshold]

    return kept


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0

    return np.logical_and(a, b).sum() / union


def dedup_by_iou(detections: list[dict], iou_threshold: float) -> list[dict]:
    '''
    Greedy by-confidence suppression -- net new step vs. a classic NMS-based model,
    since the end2end head doesn't deduplicate overlapping detections itself.
    Dedup is scoped per class so two different objects of different classes that
    happen to overlap aren't suppressed against each other.
    '''
    kept: list[dict] = []

    for det in sorted(detections, key=lambda d: d['confidence'], reverse=True):
        if any(det['class_id'] == k['class_id'] and _mask_iou(det['mask'], k['mask']) >= iou_threshold
               for k in kept):
            continue
        kept.append(det)

    return kept


def filter_masks(detections: list[dict], conf: float, alpha: float, iou_threshold: float) -> list[dict]:
    """
    Apply the full detection filter chain in order: confidence → IoU dedup → percentile.

    Args:
        detections:    Raw detections from ``results_to_masks``.
        conf:          Absolute confidence threshold.
        alpha:         Percentile threshold for ``filter_by_percentile``.
        iou_threshold: IoU overlap above which two same-class detections are deduplicated.

    Returns:
        Filtered list, typically ≤ original length.
    """
    detections = filter_by_confidence(detections, conf)
    detections = dedup_by_iou(detections, iou_threshold)
    detections = filter_by_percentile(detections, alpha)

    return detections
