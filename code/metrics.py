from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

DistanceName = Literal["euclidean", "cosine"]
DistanceFn = Callable[[Tensor, Tensor], Tensor]


def _validate_pairwise_inputs(x1: Tensor, x2: Tensor) -> None:
    """Validate two batched embedding tensors before pairwise distance calculation."""
    if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
        raise TypeError("distance inputs must be torch.Tensor objects")
    if x1.shape != x2.shape:
        raise ValueError(f"distance inputs must have the same shape, got {tuple(x1.shape)} and {tuple(x2.shape)}")
    if x1.ndim < 2:
        raise ValueError(f"expected batched embeddings with shape (batch, dim...), got {tuple(x1.shape)}")
    if x1.size(0) == 0:
        raise ValueError("empty embedding batch")


def _validate_triplet_inputs(anchors: Tensor, positives: Tensor, negatives: Tensor) -> None:
    _validate_pairwise_inputs(anchors, positives)
    _validate_pairwise_inputs(anchors, negatives)


def cosine_distance(x1: Tensor, x2: Tensor) -> Tensor:
    """Return per-sample cosine distance: 1 - cosine_similarity.

    The last dimension is treated as the embedding dimension. The returned tensor
    has one distance value per sample, matching PyTorch triplet-loss distance
    function expectations.
    """
    _validate_pairwise_inputs(x1, x2)
    return 1.0 - F.cosine_similarity(x1, x2, dim=-1, eps=1e-8)


def euclidean_distance(x1: Tensor, x2: Tensor) -> Tensor:
    """Return per-sample Euclidean distance over all non-batch dimensions."""
    _validate_pairwise_inputs(x1, x2)
    diff = x1 - x2
    return torch.linalg.vector_norm(diff.flatten(start_dim=1), ord=2, dim=1)


def get_distance_fn(distance_fn: DistanceName | DistanceFn) -> DistanceFn:
    if callable(distance_fn) and not isinstance(distance_fn, str):
        return distance_fn
    if distance_fn == "euclidean":
        return euclidean_distance
    if distance_fn == "cosine":
        return cosine_distance
    raise ValueError(f"Unsupported distance_fn: {distance_fn!r}. Expected 'euclidean', 'cosine', or a callable.")


def triplet_distance(
    anchors: Tensor,
    positives: Tensor,
    negatives: Tensor,
    distance_fn: DistanceName | DistanceFn = "cosine",
) -> tuple[Tensor, Tensor]:
    """Return anchor-positive and anchor-negative distances for a triplet batch."""
    _validate_triplet_inputs(anchors, positives, negatives)
    distance = get_distance_fn(distance_fn)
    positive_distance = distance(anchors, positives)
    negative_distance = distance(anchors, negatives)
    if positive_distance.shape != negative_distance.shape:
        raise ValueError(
            "distance_fn must return matching per-sample distances for positive and negative pairs, "
            f"got {tuple(positive_distance.shape)} and {tuple(negative_distance.shape)}"
        )
    if positive_distance.ndim == 0:
        raise ValueError("distance_fn must return one distance per sample, not a scalar for the whole batch")
    return positive_distance, negative_distance


def triplet_correct_mask(
    anchors: Tensor,
    positives: Tensor,
    negatives: Tensor,
    distance_fn: DistanceName | DistanceFn = "cosine",
    margin: float = 0.0,
) -> Tensor:
    """Return a boolean mask for correctly ranked triplets.

    With margin=0.0 this is ranking accuracy: d(anchor, positive) < d(anchor, negative).
    With margin>0.0 this is margin-satisfied accuracy: d(anchor, positive) + margin < d(anchor, negative).
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    positive_distance, negative_distance = triplet_distance(anchors, positives, negatives, distance_fn)
    return positive_distance + margin < negative_distance


def triplet_correct_predictions(
    anchors: Tensor,
    positives: Tensor,
    negatives: Tensor,
    distance_fn: DistanceName | DistanceFn = "cosine",
    margin: float = 0.0,
) -> int:
    """Count correct triplets in a batch.

    The default margin=0.0 preserves the usual top-1/ranking accuracy definition.
    Pass margin equal to the triplet-loss margin to count only zero-loss triplets.
    """
    return int(triplet_correct_mask(anchors, positives, negatives, distance_fn, margin=margin).sum().item())


def triplet_accuracy(
    anchors: Tensor,
    positives: Tensor,
    negatives: Tensor,
    distance_fn: DistanceName | DistanceFn = "cosine",
    margin: float = 0.0,
) -> float:
    """Return correct_count / batch_size for a triplet batch."""
    mask = triplet_correct_mask(anchors, positives, negatives, distance_fn, margin=margin)
    return float(mask.float().mean().item())


# Backward-compatible convenience wrappers used by existing notebooks/code.
def triplet_correct_predictions_cosine(anchors: Tensor, positives: Tensor, negatives: Tensor, margin: float = 0.0) -> int:
    return triplet_correct_predictions(anchors, positives, negatives, "cosine", margin=margin)


def triplet_distance_cosine(anchors: Tensor, positives: Tensor, negatives: Tensor) -> tuple[Tensor, Tensor]:
    return triplet_distance(anchors, positives, negatives, "cosine")


def triplet_correct_predictions_euclidean(anchors: Tensor, positives: Tensor, negatives: Tensor, margin: float = 0.0) -> int:
    return triplet_correct_predictions(anchors, positives, negatives, "euclidean", margin=margin)


def triplet_distance_euclidean(anchors: Tensor, positives: Tensor, negatives: Tensor) -> tuple[Tensor, Tensor]:
    return triplet_distance(anchors, positives, negatives, "euclidean")
