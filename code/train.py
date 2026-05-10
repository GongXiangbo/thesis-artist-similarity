from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from torch import Tensor

from metrics import DistanceName, triplet_correct_predictions, triplet_distance
from mining import BatchSemiHardNegativeMiner


NegativeMiningMode = Literal["fixed", "random", "batch_semihard"]


def _split_batch(batch) -> tuple[Tensor, Tensor, Tensor, Sequence[Any] | None, Sequence[Any] | None, Sequence[Any] | None]:
    if len(batch) not in {3, 6}:
        raise ValueError(
            "Expected a triplet batch (anchors, positives, negatives) optionally followed by "
            f"(anchor_ids, positive_ids, negative_ids), got {len(batch)} items"
        )
    anchors, positives, negatives = batch[:3]
    if len(batch) == 3:
        return anchors, positives, negatives, None, None, None
    anchor_ids, positive_ids, negative_ids = batch[3:]
    return anchors, positives, negatives, anchor_ids, positive_ids, negative_ids


def _move_batch_to_device(batch, device: torch.device | str) -> tuple[Tensor, Tensor, Tensor]:
    anchors, positives, negatives, _, _, _ = _split_batch(batch)
    return (
        anchors.to(device, non_blocking=True).float(),
        positives.to(device, non_blocking=True).float(),
        negatives.to(device, non_blocking=True).float(),
    )


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


def _id_list(ids: Sequence[Any]) -> list[Any]:
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu()
        if ids.ndim == 0:
            ids = ids.view(1)
        return [item.item() if item.ndim == 0 else tuple(item.tolist()) for item in ids]
    return list(ids)


def _encode_unique_artists(
    model,
    anchors: Tensor,
    positives: Tensor,
    negatives: Tensor,
    anchor_ids: Sequence[Any],
    positive_ids: Sequence[Any],
    negative_ids: Sequence[Any],
) -> tuple[Tensor, Tensor, Tensor, list[Any]]:
    unique_ids: list[Any] = []
    unique_tensors: list[Tensor] = []
    id_to_idx: dict[str, int] = {}

    for tensors, ids in (
        (anchors, _id_list(anchor_ids)),
        (positives, _id_list(positive_ids)),
        (negatives, _id_list(negative_ids)),
    ):
        if len(ids) != int(tensors.size(0)):
            raise ValueError("Batch ID count must match tensor batch size")
        for row_idx, artist_id in enumerate(ids):
            key = str(artist_id)
            if key in id_to_idx:
                continue
            id_to_idx[key] = len(unique_ids)
            unique_ids.append(artist_id)
            unique_tensors.append(tensors[row_idx])

    if not unique_tensors:
        raise ValueError("No candidate artists were found in the batch")

    unique_inputs = torch.stack(unique_tensors, dim=0)
    unique_embeddings = model.forward_once(unique_inputs)

    def gather(ids: Sequence[Any]) -> Tensor:
        indices = torch.tensor([id_to_idx[str(artist_id)] for artist_id in _id_list(ids)], device=unique_embeddings.device)
        return unique_embeddings.index_select(0, indices)

    anchor_embeddings = gather(anchor_ids)
    positive_embeddings = gather(positive_ids)
    return anchor_embeddings, positive_embeddings, unique_embeddings, unique_ids


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
    negative_mining: NegativeMiningMode = "fixed",
    positive_map: Mapping[Any, set[Any]] | None = None,
    mining_fallback: str = "closest_valid",
):
    """Train one epoch on triplet batches.

    Returns ``(average_loss, ranking_accuracy)`` by default for backward compatibility.
    ``ranking_accuracy`` means d(anchor, positive) < d(anchor, negative). When
    ``return_details=True``, also returns margin accuracy, which counts triplets that
    satisfy d(anchor, positive) + criterion.margin < d(anchor, negative).

    Set ``negative_mining="batch_semihard"`` to ignore the fixed negative for the
    loss and instead mine from all unique artists present in the current training
    batch. Validation should continue to call ``evaluate(...)`` on fixed triplets.
    """
    if negative_mining not in {"fixed", "random", "batch_semihard"}:
        raise ValueError("negative_mining must be one of: 'fixed', 'random', 'batch_semihard'")

    model.train()
    total_loss = 0.0
    correct = 0.0
    margin_correct = 0.0
    total = 0
    loss_weight_total = 0
    skipped_total = 0
    semi_hard_total = 0.0
    fallback_total = 0.0
    pos_dist_total = 0.0
    neg_dist_total = 0.0
    margin = _criterion_margin(criterion)
    use_batch_mining = negative_mining == "batch_semihard"
    miner = BatchSemiHardNegativeMiner(margin=margin, distance=distance_fn, fallback=mining_fallback) if use_batch_mining else None

    for batch in train_loader:
        anchors, positives, negatives, anchor_ids, positive_ids, negative_ids = _split_batch(batch)
        anchors = anchors.to(device, non_blocking=True).float()
        positives = positives.to(device, non_blocking=True).float()
        negatives = negatives.to(device, non_blocking=True).float()
        batch_size = int(anchors.size(0))

        optimizer.zero_grad(set_to_none=True)

        if use_batch_mining:
            if anchor_ids is None or positive_ids is None or negative_ids is None:
                raise ValueError(
                    "negative_mining='batch_semihard' requires batches with anchor, positive and negative artist IDs. "
                    "Use create_triplets_with_ids(...) for the training loader."
                )
            anchor_embeddings, positive_embeddings, candidate_embeddings, candidate_ids = _encode_unique_artists(
                model,
                anchors,
                positives,
                negatives,
                anchor_ids,
                positive_ids,
                negative_ids,
            )
            assert miner is not None
            mined = miner.mine(
                anchor_embeddings,
                positive_embeddings,
                candidate_embeddings,
                anchor_ids,
                positive_ids,
                candidate_ids,
                positive_map or {},
            )
            loss = mined["loss"]
            valid_mask = mined["valid_mask"]
            valid_count = int(valid_mask.sum().item())
        else:
            anchor_embeddings, positive_embeddings, negative_embeddings = model(anchors, positives, negatives)
            loss = _as_scalar_loss(criterion(anchor_embeddings, positive_embeddings, negative_embeddings))
            valid_count = batch_size
            mined = None

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
        if use_batch_mining:
            loss_weight = valid_count
            total_loss += float(loss.detach().item()) * loss_weight
            loss_weight_total += loss_weight
            total += batch_size
            skipped_total += batch_size - valid_count
            if valid_count > 0 and mined is not None:
                correct += float(mined["triplet_acc"].detach().item()) * valid_count
                margin_correct += float(mined["margin_acc"].detach().item()) * valid_count
                semi_hard_total += float(mined["semi_hard_ratio"].detach().item()) * valid_count
                fallback_total += float(mined["fallback_ratio"].detach().item()) * valid_count
                pos_dist_total += float(mined["pos_dist"][valid_mask].detach().sum().item())
                neg_dist_total += float(mined["neg_dist"][valid_mask].detach().sum().item())
            continue

        total_loss += float(loss.detach().item()) * batch_size
        loss_weight_total += batch_size
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

    metric_total = loss_weight_total if use_batch_mining else total
    if metric_total == 0:
        details = {
            "loss": 0.0,
            "acc": 0.0,
            "ranking_acc": 0.0,
            "margin_acc": 0.0,
        }
    else:
        details = _train_eval_details(total_loss, correct, margin_correct, metric_total)
    details.update(
        {
            "triplet_acc": details["ranking_acc"],
            "semi_hard_ratio": semi_hard_total / metric_total if metric_total else 0.0,
            "fallback_ratio": fallback_total / metric_total if metric_total else 0.0,
            "skipped_ratio": skipped_total / total,
            "mean_pos_dist": pos_dist_total / metric_total if metric_total else 0.0,
            "mean_neg_dist": neg_dist_total / metric_total if metric_total else 0.0,
            "negative_mining": negative_mining,
        }
    )
    if return_details:
        return details
    return details["loss"], details["ranking_acc"]
