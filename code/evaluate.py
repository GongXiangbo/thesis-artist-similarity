from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor

from metrics import DistanceName, triplet_correct_predictions


def _device_type(device: torch.device | str) -> str:
    return torch.device(device).type


def _amp_is_enabled(device: torch.device | str, enabled: bool | None) -> bool:
    if enabled is None:
        enabled = True
    return bool(enabled) and _device_type(device) == "cuda"


def _resolve_amp_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalised = str(dtype).lower().replace("torch.", "")
    if normalised in {"float16", "fp16", "half"}:
        return torch.float16
    if normalised in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError("amp_dtype must be 'float16'/'fp16' or 'bfloat16'/'bf16'")


def _autocast_context(device: torch.device | str, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    device_type = _device_type(device)
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)
    if device_type == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype, enabled=True)
    return nullcontext()


def _move_batch_to_device(batch: tuple[Tensor, Tensor, Tensor], device: torch.device | str) -> tuple[Tensor, Tensor, Tensor]:
    if len(batch) not in {3, 6}:
        raise ValueError(
            "Expected a triplet batch (anchors, positives, negatives) optionally followed by IDs, "
            f"got {len(batch)} items"
        )
    anchors, positives, negatives = batch[:3]
    return (
        anchors.to(device, non_blocking=True).float(),
        positives.to(device, non_blocking=True).float(),
        negatives.to(device, non_blocking=True).float(),
    )


def _as_scalar_loss(loss: Tensor) -> Tensor:
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


def _eval_details(total_loss: float, correct: int, margin_correct: int, total: int) -> dict[str, float]:
    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "ranking_acc": correct / total,
        "margin_acc": margin_correct / total,
    }


def evaluate(
    model,
    val_loader,
    criterion,
    device: torch.device | str,
    distance_fn: DistanceName = "cosine",
    *,
    return_details: bool = False,
    amp_enabled: bool | None = None,
    amp_dtype: torch.dtype | str = torch.float16,
):
    """Evaluate triplet batches without gradient calculation.

    Returns ``(average_loss, ranking_accuracy)`` by default for backward compatibility.
    Use ``return_details=True`` to additionally get margin accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    margin_correct = 0
    total = 0
    margin = _criterion_margin(criterion)
    use_amp = _amp_is_enabled(device, amp_enabled)
    resolved_amp_dtype = _resolve_amp_dtype(amp_dtype)

    with torch.inference_mode():
        for batch in val_loader:
            anchors, positives, negatives = _move_batch_to_device(batch, device)
            with _autocast_context(device, use_amp, resolved_amp_dtype):
                anchor_embeddings, positive_embeddings, negative_embeddings = model(anchors, positives, negatives)
                loss = _as_scalar_loss(criterion(anchor_embeddings, positive_embeddings, negative_embeddings))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite validation loss encountered: {loss.item()}")

            batch_size = int(anchor_embeddings.size(0))
            if batch_size != int(anchors.size(0)):
                raise ValueError(
                    f"Model output batch size {batch_size} does not match input batch size {int(anchors.size(0))}"
                )
            total_loss += float(loss.item()) * batch_size
            correct += triplet_correct_predictions(anchor_embeddings, positive_embeddings, negative_embeddings, distance_fn, margin=0.0)
            margin_correct += triplet_correct_predictions(
                anchor_embeddings, positive_embeddings, negative_embeddings, distance_fn, margin=margin
            )
            total += batch_size

    if total == 0:
        raise ValueError("val_loader produced zero samples")

    details = _eval_details(total_loss, correct, margin_correct, total)
    if return_details:
        return details
    return details["loss"], details["ranking_acc"]
