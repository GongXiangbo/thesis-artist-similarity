from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from metrics import DistanceName, triplet_correct_predictions, triplet_distance


def _split_batch(batch) -> tuple[Tensor, Tensor, Tensor]:
    if len(batch) != 3:
        raise ValueError(f"Expected a fixed triplet batch (anchors, positives, negatives), got {len(batch)} items")
    anchors, positives, negatives = batch
    return anchors, positives, negatives


def _as_scalar_loss(loss: Tensor) -> Tensor:
    """Accept scalar losses and reduction='none' losses, always returning a scalar mean."""
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"criterion must return a torch.Tensor, got {type(loss).__name__}")
    if loss.ndim == 0:
        return loss
    if loss.numel() == 0:
        raise ValueError("criterion returned an empty loss tensor")
    return loss.mean()


def _criterion_margin(criterion: Any) -> float:
    margin = getattr(criterion, "margin", 0.0)
    try:
        return float(margin)
    except (TypeError, ValueError):
        return 0.0


def _train_eval_details(total_loss: float, correct: float, margin_correct: float, total: int) -> dict[str, float]:
    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "ranking_acc": correct / total,
        "margin_acc": margin_correct / total,
    }


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device: torch.device | str,
    distance_fn: DistanceName = "cosine",
    grad_clip: float | None = 1.0,
    *,
    return_details: bool = False,
):
    """Train one epoch on triplet batches.

    ``ranking_accuracy`` means d(anchor, positive) < d(anchor, negative). The
    negative is always the artist stored in the triplet CSV.
    """
    model.train()
    total_loss = 0.0
    correct = 0.0
    margin_correct = 0.0
    total = 0
    pos_dist_total = 0.0
    neg_dist_total = 0.0
    margin = _criterion_margin(criterion)

    for batch in train_loader:
        anchors, positives, negatives = _split_batch(batch)
        anchors = anchors.to(device, non_blocking=True).float()
        positives = positives.to(device, non_blocking=True).float()
        negatives = negatives.to(device, non_blocking=True).float()
        batch_size = int(anchors.size(0))

        optimizer.zero_grad(set_to_none=True)

        anchor_embeddings, positive_embeddings, negative_embeddings = model(anchors, positives, negatives)
        loss = _as_scalar_loss(criterion(anchor_embeddings, positive_embeddings, negative_embeddings))

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss encountered: {loss.item()}")

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        if int(anchor_embeddings.size(0)) != batch_size:
            raise ValueError(
                f"Model output batch size {int(anchor_embeddings.size(0))} does not match input batch size {batch_size}"
            )

        total_loss += float(loss.detach().item()) * batch_size
        correct += triplet_correct_predictions(
            anchor_embeddings.detach(), positive_embeddings.detach(), negative_embeddings.detach(), distance_fn, margin=0.0
        )
        margin_correct += triplet_correct_predictions(
            anchor_embeddings.detach(), positive_embeddings.detach(), negative_embeddings.detach(), distance_fn, margin=margin
        )
        positive_distance, negative_distance = triplet_distance(
            anchor_embeddings.detach(),
            positive_embeddings.detach(),
            negative_embeddings.detach(),
            distance_fn,
        )
        pos_dist_total += float(positive_distance.sum().item())
        neg_dist_total += float(negative_distance.sum().item())
        total += batch_size

    if total == 0:
        raise ValueError("train_loader produced zero samples")

    details = _train_eval_details(total_loss, correct, margin_correct, total)
    details.update(
        {
            "triplet_acc": details["ranking_acc"],
            "mean_pos_dist": pos_dist_total / total,
            "mean_neg_dist": neg_dist_total / total,
        }
    )
    if return_details:
        return details
    return details["loss"], details["ranking_acc"]
