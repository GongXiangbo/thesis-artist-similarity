from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from mining import BatchSemiHardNegativeMiner


def unit_from_cos(cosine: float) -> torch.Tensor:
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return torch.tensor([cosine, sine], dtype=torch.float32)


def assert_close(actual: float, expected: float, tol: float = 1e-5) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"Expected {expected:.6f}, got {actual:.6f}")


def test_semihard_selection_and_backward() -> None:
    raw_anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    raw_positive = unit_from_cos(0.90).view(1, 2).requires_grad_(True)
    raw_candidates = torch.stack(
        [
            torch.tensor([1.0, 0.0]),
            unit_from_cos(0.90),
            unit_from_cos(0.80),
            unit_from_cos(0.70),
        ],
        dim=0,
    ).requires_grad_(True)

    miner = BatchSemiHardNegativeMiner(margin=0.5)
    out = miner.mine(
        F.normalize(raw_anchor, dim=1),
        F.normalize(raw_positive, dim=1),
        F.normalize(raw_candidates, dim=1),
        ["a"],
        ["p"],
        ["a", "p", "n_close", "n_far"],
        {"a": {"p"}},
    )

    assert out["selected_negative_ids"] == ["n_close"]
    assert bool(out["valid_mask"][0])
    assert_close(float(out["pos_dist"][0]), 0.10)
    assert_close(float(out["neg_dist"][0]), 0.20)
    assert_close(float(out["semi_hard_ratio"]), 1.0)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert raw_anchor.grad is not None and torch.isfinite(raw_anchor.grad).all()
    assert raw_positive.grad is not None and torch.isfinite(raw_positive.grad).all()
    assert raw_candidates.grad is not None and torch.isfinite(raw_candidates.grad).all()


def test_farther_fallback_when_no_semihard() -> None:
    anchor = torch.tensor([[1.0, 0.0]])
    positive = unit_from_cos(0.90).view(1, 2)
    candidates = torch.stack([torch.tensor([1.0, 0.0]), positive[0], unit_from_cos(0.20)], dim=0)

    miner = BatchSemiHardNegativeMiner(margin=0.5)
    out = miner.mine(anchor, positive, candidates, ["a"], ["p"], ["a", "p", "n_far"], {"a": {"p"}})

    assert out["selected_negative_ids"] == ["n_far"]
    assert_close(float(out["semi_hard_ratio"]), 0.0)
    assert_close(float(out["fallback_ratio"]), 1.0)
    assert float(out["neg_dist"][0]) > float(out["pos_dist"][0]) + 0.5


def test_closest_valid_fallback_when_all_valid_are_closer_than_positive() -> None:
    anchor = torch.tensor([[1.0, 0.0]])
    positive = unit_from_cos(0.40).view(1, 2)
    candidates = torch.stack([torch.tensor([1.0, 0.0]), positive[0], unit_from_cos(0.80)], dim=0)

    miner = BatchSemiHardNegativeMiner(margin=0.5)
    out = miner.mine(anchor, positive, candidates, ["a"], ["p"], ["a", "p", "n_close"], {"a": {"p"}})

    assert out["selected_negative_ids"] == ["n_close"]
    assert_close(float(out["fallback_ratio"]), 1.0)
    assert float(out["neg_dist"][0]) < float(out["pos_dist"][0])
    assert float(out["loss"]) > 0.0


def test_known_positives_and_self_are_excluded() -> None:
    anchor = torch.tensor([[1.0, 0.0]])
    positive = unit_from_cos(0.90).view(1, 2)
    candidates = torch.stack(
        [
            torch.tensor([1.0, 0.0]),
            positive[0],
            unit_from_cos(0.89),
            unit_from_cos(0.70),
        ],
        dim=0,
    )

    miner = BatchSemiHardNegativeMiner(margin=0.5)
    out = miner.mine(
        anchor,
        positive,
        candidates,
        ["a"],
        ["p"],
        ["a", "p", "known_positive", "valid_negative"],
        {"a": {"p", "known_positive"}},
    )

    assert out["selected_negative_ids"] == ["valid_negative"]


def test_skipped_anchor_and_all_skipped_backward() -> None:
    anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    positive = torch.stack([unit_from_cos(0.90), unit_from_cos(0.80)], dim=0).requires_grad_(True)
    candidates = torch.stack([torch.tensor([1.0, 0.0]), unit_from_cos(0.90), unit_from_cos(0.80)], dim=0).requires_grad_(True)

    miner = BatchSemiHardNegativeMiner(margin=0.5)
    out = miner.mine(
        F.normalize(anchor, dim=1),
        F.normalize(positive, dim=1),
        F.normalize(candidates, dim=1),
        ["a", "b"],
        ["p_a", "p_b"],
        ["a", "p_a", "p_b"],
        {"a": {"p_a", "p_b"}, "b": {"a", "p_a", "p_b"}},
    )

    assert out["valid_mask"].tolist() == [False, False]
    assert_close(float(out["skipped_ratio"]), 1.0)
    assert_close(float(out["loss"]), 0.0)
    out["loss"].backward()
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()
    assert positive.grad is not None and torch.isfinite(positive.grad).all()
    assert candidates.grad is not None and torch.isfinite(candidates.grad).all()


if __name__ == "__main__":
    test_semihard_selection_and_backward()
    test_farther_fallback_when_no_semihard()
    test_closest_valid_fallback_when_all_valid_are_closer_than_positive()
    test_known_positives_and_self_are_excluded()
    test_skipped_anchor_and_all_skipped_backward()
    print("All BatchSemiHardNegativeMiner sanity checks passed.")
