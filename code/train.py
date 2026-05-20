from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from metrics import DistanceName, triplet_correct_predictions, triplet_distance
from mining import BatchSemiHardNegativeMiner


NegativeMiningMode = Literal["fixed", "random", "batch_semihard", "memory_bank_semihard"]


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


def _distance_pair(x1: Tensor, x2: Tensor, distance_fn: DistanceName) -> Tensor:
    if distance_fn == "cosine":
        x1 = F.normalize(x1, dim=1, eps=1e-8)
        x2 = F.normalize(x2, dim=1, eps=1e-8)
        return (1.0 - torch.sum(x1 * x2, dim=1)).clamp_min(0.0)
    if distance_fn == "euclidean":
        return torch.linalg.vector_norm(x1 - x2, ord=2, dim=1)
    raise ValueError(f"Unsupported distance_fn: {distance_fn!r}")


def _stack_memory_bank_tensors(memory_bank_tensors: Sequence[Tensor]) -> Tensor:
    if not memory_bank_tensors:
        raise ValueError("memory_bank_tensors is empty")
    shapes = {tuple(tensor.shape) for tensor in memory_bank_tensors}
    if len(shapes) != 1:
        raise ValueError(f"All memory-bank tensors must have the same shape, got {sorted(shapes)}")
    return torch.stack([tensor.float().cpu() for tensor in memory_bank_tensors], dim=0)


def _encode_memory_bank(
    model,
    memory_bank_tensors: Sequence[Tensor],
    device: torch.device | str,
    *,
    batch_size: int,
) -> Tensor:
    """Encode all training artists once at the start of an epoch.

    The memory bank is used only to *select* hard negatives. Once a negative ID is
    selected, the selected negative is forwarded again with gradients enabled.
    """
    memory_inputs = _stack_memory_bank_tensors(memory_bank_tensors)
    encoded_chunks: list[Tensor] = []
    was_training = bool(model.training)
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(memory_inputs.size(0)), batch_size):
            chunk = memory_inputs[start : start + batch_size].to(device, non_blocking=True).float()
            encoded_chunks.append(model.forward_once(chunk).detach())
    if was_training:
        model.train()
    return torch.cat(encoded_chunks, dim=0)


def _selected_memory_tensors(
    selected_ids: Sequence[Any],
    valid_mask: Tensor,
    memory_tensor_by_id: Mapping[str, Tensor],
) -> tuple[list[int], Tensor | None]:
    valid_positions: list[int] = []
    selected_tensors: list[Tensor] = []
    selected_list = _id_list(selected_ids)
    valid_mask_cpu = valid_mask.detach().cpu().bool().tolist()
    for row_idx, is_valid in enumerate(valid_mask_cpu):
        if not is_valid:
            continue
        selected_id = selected_list[row_idx]
        if selected_id is None or str(selected_id) == "-1":
            continue
        tensor = memory_tensor_by_id.get(str(selected_id))
        if tensor is None:
            continue
        valid_positions.append(row_idx)
        selected_tensors.append(tensor.float())
    if not selected_tensors:
        return [], None
    return valid_positions, torch.stack(selected_tensors, dim=0)


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
    memory_bank_ids: Sequence[Any] | None = None,
    memory_bank_tensors: Sequence[Tensor] | None = None,
    memory_bank_batch_size: int | None = None,
):
    """Train one epoch on triplet batches.

    ``ranking_accuracy`` means d(anchor, positive) < d(anchor, negative). When
    ``return_details=True``, the return dict also includes margin accuracy and
    mining diagnostics.

    Negative mining modes:
    - ``fixed`` / ``random``: use the negative already stored in each triplet.
    - ``batch_semihard``: mine from unique artists in the current batch.
    - ``memory_bank_semihard``: priority-1 mode. At the beginning of each epoch,
      encode all training artists into a memory bank, select hard negatives from
      that global pool, then forward the selected negatives again with gradients.
    """
    valid_modes = {"fixed", "random", "batch_semihard", "memory_bank_semihard"}
    if negative_mining not in valid_modes:
        raise ValueError(f"negative_mining must be one of: {sorted(valid_modes)}")

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
    use_memory_bank_mining = negative_mining == "memory_bank_semihard"
    use_mining = use_batch_mining or use_memory_bank_mining
    miner = BatchSemiHardNegativeMiner(margin=margin, distance=distance_fn, fallback=mining_fallback) if use_mining else None

    memory_embeddings: Tensor | None = None
    memory_ids: Sequence[Any] | None = None
    memory_tensor_by_id: dict[str, Tensor] = {}
    if use_memory_bank_mining:
        if memory_bank_ids is None or memory_bank_tensors is None:
            raise ValueError(
                "negative_mining='memory_bank_semihard' requires memory_bank_ids and memory_bank_tensors. "
                "Use create_artist_memory_bank(train_df, artist_averages)."
            )
        if len(memory_bank_ids) != len(memory_bank_tensors):
            raise ValueError("memory_bank_ids and memory_bank_tensors must have the same length")
        memory_ids = [str(item) for item in _id_list(memory_bank_ids)]
        memory_tensor_by_id = {str(artist_id): tensor.float().cpu() for artist_id, tensor in zip(memory_ids, memory_bank_tensors)}
        encode_bs = int(memory_bank_batch_size or max(1, getattr(train_loader, "batch_size", 128) or 128))
        memory_embeddings = _encode_memory_bank(model, memory_bank_tensors, device, batch_size=encode_bs)
        if int(memory_embeddings.size(0)) != len(memory_ids):
            raise RuntimeError("Encoded memory bank size does not match memory_bank_ids")
        model.train()

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

        elif use_memory_bank_mining:
            if anchor_ids is None or positive_ids is None:
                raise ValueError(
                    "negative_mining='memory_bank_semihard' requires batches with anchor and positive artist IDs. "
                    "Use create_triplets_with_ids(...) for the training loader."
                )
            if memory_embeddings is None or memory_ids is None:
                raise RuntimeError("Memory bank was not initialized")

            anchor_embeddings = model.forward_once(anchors)
            positive_embeddings = model.forward_once(positives)
            assert miner is not None
            with torch.no_grad():
                mined = miner.mine(
                    anchor_embeddings.detach(),
                    positive_embeddings.detach(),
                    memory_embeddings,
                    anchor_ids,
                    positive_ids,
                    memory_ids,
                    positive_map or {},
                )
            valid_mask = mined["valid_mask"].to(anchor_embeddings.device)
            selected_positions, selected_negative_inputs = _selected_memory_tensors(
                mined["selected_negative_ids"],
                valid_mask,
                memory_tensor_by_id,
            )
            valid_count = len(selected_positions)
            if valid_count > 0 and selected_negative_inputs is not None:
                row_indices = torch.tensor(selected_positions, device=anchor_embeddings.device, dtype=torch.long)
                selected_anchor_embeddings = anchor_embeddings.index_select(0, row_indices)
                selected_positive_embeddings = positive_embeddings.index_select(0, row_indices)
                selected_negative_inputs = selected_negative_inputs.to(device, non_blocking=True).float()
                selected_negative_embeddings = model.forward_once(selected_negative_inputs)

                positive_distance = _distance_pair(selected_anchor_embeddings, selected_positive_embeddings, distance_fn)
                negative_distance = _distance_pair(selected_anchor_embeddings, selected_negative_embeddings, distance_fn)
                losses = F.relu(positive_distance - negative_distance + margin)
                loss = losses.mean()
                triplet_acc = (positive_distance < negative_distance).float().mean()
                margin_acc = (positive_distance + margin < negative_distance).float().mean()
            else:
                # Keep the graph connected so backward() is still valid if an
                # unusually strict exclusion map leaves no candidate for a batch.
                loss = (anchor_embeddings.sum() + positive_embeddings.sum()) * 0.0
                positive_distance = torch.empty(0, device=anchor_embeddings.device, dtype=anchor_embeddings.dtype)
                negative_distance = torch.empty(0, device=anchor_embeddings.device, dtype=anchor_embeddings.dtype)
                triplet_acc = torch.zeros((), device=anchor_embeddings.device, dtype=anchor_embeddings.dtype)
                margin_acc = torch.zeros((), device=anchor_embeddings.device, dtype=anchor_embeddings.dtype)

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

        if use_memory_bank_mining:
            loss_weight = valid_count
            total_loss += float(loss.detach().item()) * loss_weight
            loss_weight_total += loss_weight
            total += batch_size
            skipped_total += batch_size - valid_count
            if valid_count > 0 and mined is not None:
                correct += float(triplet_acc.detach().item()) * valid_count
                margin_correct += float(margin_acc.detach().item()) * valid_count
                semi_hard_total += float(mined["semi_hard_ratio"].detach().item()) * valid_count
                fallback_total += float(mined["fallback_ratio"].detach().item()) * valid_count
                pos_dist_total += float(positive_distance.detach().sum().item())
                neg_dist_total += float(negative_distance.detach().sum().item())
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

    metric_total = loss_weight_total if use_mining else total
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
            "memory_bank_size": len(memory_ids) if memory_ids is not None else 0,
        }
    )
    if return_details:
        return details
    return details["loss"], details["ranking_acc"]
