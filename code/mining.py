from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

MiningDistance = Literal["cosine", "euclidean"]


def _ids_to_list(ids: Tensor | Sequence[Any]) -> list[Any]:
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu()
        if ids.ndim == 0:
            ids = ids.view(1)
        return [item.item() if item.ndim == 0 else tuple(item.tolist()) for item in ids]
    return list(ids)


def _ids_to_keys(ids: Tensor | Sequence[Any]) -> list[str]:
    return [str(item) for item in _ids_to_list(ids)]


def _lookup_positive_set(positive_map: Mapping[Any, set[Any]] | None, anchor_id: Any, anchor_key: str) -> set[str]:
    if not positive_map:
        return set()
    positives = positive_map.get(anchor_key)
    if positives is None:
        positives = positive_map.get(anchor_id, set())
    return {str(item) for item in positives}


def _selected_ids(
    candidate_ids: Tensor | Sequence[Any],
    selected_indices: Tensor,
    valid_mask: Tensor,
) -> Tensor | list[Any | None]:
    if isinstance(candidate_ids, torch.Tensor):
        if candidate_ids.ndim != 1:
            raise ValueError("candidate_ids tensor must be 1D")
        out = torch.full(
            (selected_indices.numel(),),
            -1,
            dtype=candidate_ids.dtype,
            device=selected_indices.device,
        )
        if valid_mask.any():
            out[valid_mask] = candidate_ids.to(selected_indices.device)[selected_indices[valid_mask]]
        return out

    candidate_list = _ids_to_list(candidate_ids)
    return [
        candidate_list[int(selected_indices[idx].item())] if bool(valid_mask[idx].item()) else None
        for idx in range(selected_indices.numel())
    ]


class BatchSemiHardNegativeMiner:
    """Mine one negative per anchor-positive pair from the current batch.

    The miner treats all unique artists in the batch as candidate negatives.
    Known positives from the fold-local ``positive_map`` are masked out to avoid
    false negatives. Selection prefers the closest semi-hard negative, then the
    closest valid negative outside the positive distance, then the closest valid
    negative overall.
    """

    def __init__(
        self,
        margin: float = 0.5,
        distance: MiningDistance = "cosine",
        fallback: str = "closest_valid",
    ) -> None:
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if distance not in {"cosine", "euclidean"}:
            raise ValueError("distance must be 'cosine' or 'euclidean'")
        if fallback != "closest_valid":
            raise ValueError("Only fallback='closest_valid' is supported")
        self.margin = float(margin)
        self.distance = distance
        self.fallback = fallback

    def _pair_distance(self, x1: Tensor, x2: Tensor) -> Tensor:
        if self.distance == "cosine":
            x1 = F.normalize(x1, dim=1, eps=1e-8)
            x2 = F.normalize(x2, dim=1, eps=1e-8)
            return (1.0 - torch.sum(x1 * x2, dim=1)).clamp_min(0.0)
        return torch.linalg.vector_norm(x1 - x2, ord=2, dim=1)

    def _distance_matrix(self, anchors: Tensor, candidates: Tensor) -> Tensor:
        if self.distance == "cosine":
            anchors = F.normalize(anchors, dim=1, eps=1e-8)
            candidates = F.normalize(candidates, dim=1, eps=1e-8)
            cosine_sim = (anchors @ candidates.T).clamp(min=-1.0, max=1.0)
            return (1.0 - cosine_sim).clamp_min(0.0)
        return torch.cdist(anchors, candidates, p=2)

    def mine(
        self,
        anchor_emb: Tensor,
        positive_emb: Tensor,
        candidate_emb: Tensor,
        anchor_ids: Tensor | Sequence[Any],
        positive_ids: Tensor | Sequence[Any],
        candidate_ids: Tensor | Sequence[Any],
        positive_map: Mapping[Any, set[Any]] | None,
    ) -> dict[str, Any]:
        if anchor_emb.ndim != 2 or positive_emb.ndim != 2 or candidate_emb.ndim != 2:
            raise ValueError("anchor_emb, positive_emb and candidate_emb must be 2D tensors")
        if anchor_emb.shape != positive_emb.shape:
            raise ValueError(
                f"anchor_emb and positive_emb must have the same shape, got {anchor_emb.shape} and {positive_emb.shape}"
            )
        if anchor_emb.size(1) != candidate_emb.size(1):
            raise ValueError("candidate embedding dimension must match anchor embedding dimension")

        batch_size = int(anchor_emb.size(0))
        candidate_count = int(candidate_emb.size(0))
        device = anchor_emb.device
        if batch_size == 0:
            raise ValueError("empty anchor batch")
        if candidate_count == 0:
            raise ValueError("empty candidate batch")

        anchor_id_values = _ids_to_list(anchor_ids)
        positive_id_keys = _ids_to_keys(positive_ids)
        candidate_id_keys = _ids_to_keys(candidate_ids)
        if len(anchor_id_values) != batch_size or len(positive_id_keys) != batch_size:
            raise ValueError("anchor_ids and positive_ids must match the anchor batch size")
        if len(candidate_id_keys) != candidate_count:
            raise ValueError("candidate_ids must match candidate_emb rows")

        pos_dist = self._pair_distance(anchor_emb, positive_emb)
        distance_matrix = self._distance_matrix(anchor_emb, candidate_emb)

        valid_negative_mask = torch.ones((batch_size, candidate_count), dtype=torch.bool, device=device)
        for row_idx, anchor_id in enumerate(anchor_id_values):
            anchor_key = str(anchor_id)
            known_positive_keys = _lookup_positive_set(positive_map, anchor_id, anchor_key)
            # Always mask the paired positive, even if a sparse map missed it.
            known_positive_keys.add(positive_id_keys[row_idx])
            for col_idx, candidate_key in enumerate(candidate_id_keys):
                if candidate_key == anchor_key or candidate_key in known_positive_keys:
                    valid_negative_mask[row_idx, col_idx] = False

        inf = torch.tensor(float("inf"), device=device, dtype=distance_matrix.dtype)
        with torch.no_grad():
            pos_dist_col = pos_dist.detach().unsqueeze(1)
            dist_detached = distance_matrix.detach()

            semi_hard_mask = (
                valid_negative_mask
                & (dist_detached > pos_dist_col)
                & (dist_detached < pos_dist_col + self.margin)
            )
            farther_valid_mask = valid_negative_mask & (dist_detached > pos_dist_col)

            semi_scores = dist_detached.masked_fill(~semi_hard_mask, inf)
            farther_scores = dist_detached.masked_fill(~farther_valid_mask, inf)
            valid_scores = dist_detached.masked_fill(~valid_negative_mask, inf)

            semi_idx = semi_scores.argmin(dim=1)
            farther_idx = farther_scores.argmin(dim=1)
            valid_idx = valid_scores.argmin(dim=1)

            has_semi = semi_hard_mask.any(dim=1)
            has_farther = farther_valid_mask.any(dim=1)
            has_valid = valid_negative_mask.any(dim=1)

            selected_idx = torch.where(has_semi, semi_idx, torch.where(has_farther, farther_idx, valid_idx))
            valid_mask = has_valid
            fallback_mask = valid_mask & ~has_semi
            safe_idx = selected_idx.clamp_min(0)

        neg_dist = torch.full_like(pos_dist, float("nan"))
        selected_negative_emb = candidate_emb[safe_idx] * valid_mask.to(candidate_emb.dtype).unsqueeze(1)
        if valid_mask.any():
            neg_dist[valid_mask] = distance_matrix[valid_mask, safe_idx[valid_mask]]
            selected_negative_emb[valid_mask] = candidate_emb[safe_idx[valid_mask]]

        if valid_mask.any():
            selected_pos_dist = pos_dist[valid_mask]
            selected_neg_dist = neg_dist[valid_mask]
            losses = F.relu(selected_pos_dist - selected_neg_dist + self.margin)
            loss = losses.mean()
            triplet_acc = (selected_pos_dist < selected_neg_dist).float().mean()
            margin_acc = (selected_pos_dist + self.margin < selected_neg_dist).float().mean()
            selected_count = valid_mask.float().sum()
            semi_hard_ratio = (has_semi & valid_mask).float().sum() / selected_count
            fallback_ratio = fallback_mask.float().sum() / selected_count
        else:
            loss = (anchor_emb.sum() + positive_emb.sum() + candidate_emb.sum()) * 0.0
            triplet_acc = torch.zeros((), device=device, dtype=anchor_emb.dtype)
            margin_acc = torch.zeros((), device=device, dtype=anchor_emb.dtype)
            semi_hard_ratio = torch.zeros((), device=device, dtype=anchor_emb.dtype)
            fallback_ratio = torch.zeros((), device=device, dtype=anchor_emb.dtype)

        skipped_ratio = (~valid_mask).float().mean()

        return {
            "loss": loss,
            "selected_negative_emb": selected_negative_emb,
            "selected_negative_ids": _selected_ids(candidate_ids, safe_idx, valid_mask),
            "valid_mask": valid_mask,
            "pos_dist": pos_dist,
            "neg_dist": neg_dist,
            "semi_hard_ratio": semi_hard_ratio,
            "fallback_ratio": fallback_ratio,
            "skipped_ratio": skipped_ratio,
            "triplet_acc": triplet_acc,
            "margin_acc": margin_acc,
        }
