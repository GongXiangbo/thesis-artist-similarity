from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from dataset import (
    build_positive_map,
    create_dataloaders_from_triplet_lists,
    create_triplets,
    create_triplets_with_ids,
)
from evaluate import evaluate
from metrics import DistanceName, cosine_distance
from train import NegativeMiningMode, train


TRIPLET_COLUMNS = ("anchor", "positive", "negative")


@dataclass(frozen=True)
class FoldTrainingConfig:
    model_name: str
    model_class: type[nn.Module]
    model_kwargs: dict[str, Any]
    output_dir: Path
    device: torch.device | str
    batch_size: int = 128
    num_workers: int = 0
    num_epochs: int = 30
    learning_rate: float = 2e-4
    weight_decay: float = 1e-6
    early_stopping_patience: int | None = 8
    distance_fn: DistanceName = "cosine"
    negative_mining: NegativeMiningMode = "batch_semihard"
    mining_fallback: str = "closest_valid"
    grad_clip: float | None = 1.0


def make_anchor_group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 3407,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    """Create 5-fold CV splits grouped by anchor artist.

    Validation folds contain anchors that never appear as anchors in the matching
    training fold. Positive/negative artists may still overlap, which keeps the
    sparse triplet graph usable while preserving the previous notebook protocol.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    df = df.copy().reset_index(drop=True)
    for col in TRIPLET_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required triplet column: {col}")
        df[col] = df[col].astype(str)

    anchors = np.array(sorted(df["anchor"].unique()), dtype=object)
    if len(anchors) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique anchors for {n_splits}-fold CV, "
            f"but only found {len(anchors)}."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(anchors)
    anchor_folds = np.array_split(anchors, n_splits)

    folds = []
    for fold_id, val_anchor_array in enumerate(anchor_folds, start=1):
        val_anchors = set(map(str, val_anchor_array.tolist()))
        val_mask = df["anchor"].isin(val_anchors)

        train_df = df.loc[~val_mask].reset_index(drop=True)
        val_df = df.loc[val_mask].reset_index(drop=True)
        if train_df.empty or val_df.empty:
            raise ValueError(f"Fold {fold_id} produced an empty train or validation split")

        train_anchor_set = set(train_df["anchor"])
        val_anchor_set = set(val_df["anchor"])
        train_artist_set = set(train_df[list(TRIPLET_COLUMNS)].to_numpy().reshape(-1))
        val_artist_set = set(val_df[list(TRIPLET_COLUMNS)].to_numpy().reshape(-1))

        stats = {
            "fold": fold_id,
            "strategy": "anchor_group_kfold",
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "train_anchors": len(train_anchor_set),
            "val_anchors": len(val_anchor_set),
            "anchor_overlap": bool(train_anchor_set & val_anchor_set),
            "train_artists": len(train_artist_set),
            "val_artists": len(val_artist_set),
            "artist_overlap": bool(train_artist_set & val_artist_set),
        }
        folds.append((fold_id, train_df, val_df, stats))

    return folds


def make_cosine_triplet_criterion(margin: float):
    return nn.TripletMarginWithDistanceLoss(
        distance_function=cosine_distance,
        margin=margin,
        swap=True,
    )


def _make_criterion(margin: float, distance_fn: DistanceName):
    if distance_fn == "cosine":
        return make_cosine_triplet_criterion(margin)
    if distance_fn == "euclidean":
        return nn.TripletMarginLoss(margin=margin, swap=True)
    raise ValueError(f"Unsupported distance_fn: {distance_fn!r}")


def _state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def run_one_fold_margin(
    fold_id: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    margin: float,
    artist_averages: dict[str, torch.Tensor],
    config: FoldTrainingConfig,
) -> dict[str, Any]:
    if config.negative_mining == "batch_semihard":
        train_triplets = create_triplets_with_ids(train_df, artist_averages)
        positive_map = build_positive_map(train_df, symmetric=True)
    else:
        train_triplets = create_triplets(train_df, artist_averages)
        positive_map = None
    val_triplets = create_triplets(val_df, artist_averages)

    if not train_triplets or not val_triplets:
        raise RuntimeError(f"Fold {fold_id}, margin={margin}: empty train/validation triplets after tensor creation.")

    train_loader, val_loader = create_dataloaders_from_triplet_lists(
        train_triplets,
        val_triplets,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = config.model_class(**config.model_kwargs).to(config.device)
    criterion = _make_criterion(margin, config.distance_fn)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=3,
    )

    best_val_acc = -1.0
    best_val_margin_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train(
            model,
            train_loader,
            criterion,
            optimizer,
            config.device,
            distance_fn=config.distance_fn,
            grad_clip=config.grad_clip,
            return_details=True,
            negative_mining=config.negative_mining,
            positive_map=positive_map,
            mining_fallback=config.mining_fallback,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            config.device,
            distance_fn=config.distance_fn,
            return_details=True,
        )
        scheduler.step(val_metrics["loss"])

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "model": config.model_name,
            "fold": fold_id,
            "margin": margin,
            "negative_mining": config.negative_mining,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["ranking_acc"],
            "train_triplet_acc": train_metrics["ranking_acc"],
            "train_margin_acc": train_metrics["margin_acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["ranking_acc"],
            "val_margin_acc": val_metrics["margin_acc"],
            "semi_hard_ratio": train_metrics.get("semi_hard_ratio", 0.0),
            "fallback_ratio": train_metrics.get("fallback_ratio", 0.0),
            "skipped_ratio": train_metrics.get("skipped_ratio", 0.0),
            "mean_pos_dist": train_metrics.get("mean_pos_dist", 0.0),
            "mean_neg_dist": train_metrics.get("mean_neg_dist", 0.0),
            "lr": lr,
        }
        history.append(row)

        improved = (
            (val_metrics["ranking_acc"] > best_val_acc)
            or (
                val_metrics["ranking_acc"] == best_val_acc
                and val_metrics["loss"] < best_val_loss
            )
        )
        if improved:
            best_val_acc = val_metrics["ranking_acc"]
            best_val_margin_acc = val_metrics["margin_acc"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = _state_dict_to_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        mining_bits = ""
        if config.negative_mining == "batch_semihard":
            mining_bits = (
                f" | semi={train_metrics['semi_hard_ratio']:.1%}"
                f" | fallback={train_metrics['fallback_ratio']:.1%}"
                f" | skipped={train_metrics['skipped_ratio']:.1%}"
            )
        print(
            f"fold={fold_id} | mining={config.negative_mining} | margin={margin:.2f} | "
            f"epoch={epoch:03d}/{config.num_epochs} | "
            f"train_loss={train_metrics['loss']:.5f} | train_acc={train_metrics['ranking_acc']:.2%} | "
            f"val_loss={val_metrics['loss']:.5f} | val_acc={val_metrics['ranking_acc']:.2%} | "
            f"val_margin_acc={val_metrics['margin_acc']:.2%} | lr={lr:.2e}{mining_bits}"
        )

        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: no validation ranking-accuracy improvement "
                f"for {config.early_stopping_patience} epochs."
            )
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold_id}, margin={margin}: no best model state was captured.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / f"{config.model_name}_margin_{margin:.2f}_fold_{fold_id}_best.pt"
    torch.save(best_state, checkpoint_path)

    history_df = pd.DataFrame(history)
    history_path = config.output_dir / f"{config.model_name}_margin_{margin:.2f}_fold_{fold_id}_history.csv"
    history_df.to_csv(history_path, index=False)

    return {
        "model": config.model_name,
        "fold": fold_id,
        "margin": margin,
        "negative_mining": config.negative_mining,
        "best_epoch": best_epoch,
        "epochs_ran": len(history_df),
        "best_val_acc": best_val_acc,
        "best_val_margin_acc": best_val_margin_acc,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "history": history_df,
        "val_triplets": val_triplets,
    }


def _history_mean(item: dict[str, Any], column: str) -> float:
    history = item.get("history")
    if not isinstance(history, pd.DataFrame) or column not in history.columns or history.empty:
        return float("nan")
    return float(history[column].mean())


def summarize_cv_results(
    results: list[dict[str, Any]],
    output_dir: Path,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_summary_df = pd.DataFrame(
        [
            {
                "model": item["model"],
                "margin": item["margin"],
                "negative_mining": item.get("negative_mining", "fixed"),
                "fold": item["fold"],
                "best_epoch": item["best_epoch"],
                "epochs_ran": item["epochs_ran"],
                "best_val_acc": item["best_val_acc"],
                "best_val_margin_acc": item["best_val_margin_acc"],
                "best_val_loss": item["best_val_loss"],
                "mean_semi_hard_ratio": _history_mean(item, "semi_hard_ratio"),
                "mean_fallback_ratio": _history_mean(item, "fallback_ratio"),
                "mean_skipped_ratio": _history_mean(item, "skipped_ratio"),
                "checkpoint_path": item["checkpoint_path"],
                "history_path": item["history_path"],
            }
            for item in results
        ]
    ).sort_values(["margin", "fold"]).reset_index(drop=True)

    margin_summary_df = (
        fold_summary_df
        .groupby(["model", "negative_mining", "margin"], as_index=False)
        .agg(
            mean_best_val_acc=("best_val_acc", "mean"),
            std_best_val_acc=("best_val_acc", "std"),
            mean_best_val_margin_acc=("best_val_margin_acc", "mean"),
            std_best_val_margin_acc=("best_val_margin_acc", "std"),
            mean_best_val_loss=("best_val_loss", "mean"),
            std_best_val_loss=("best_val_loss", "std"),
            mean_best_epoch=("best_epoch", "mean"),
            mean_epochs_ran=("epochs_ran", "mean"),
            mean_semi_hard_ratio=("mean_semi_hard_ratio", "mean"),
            mean_fallback_ratio=("mean_fallback_ratio", "mean"),
            mean_skipped_ratio=("mean_skipped_ratio", "mean"),
        )
        .sort_values(
            ["mean_best_val_acc", "mean_best_val_loss"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_summary_path = output_dir / f"{model_name}_5fold_fold_summary.csv"
    margin_summary_path = output_dir / f"{model_name}_5fold_margin_summary.csv"
    fold_summary_df.to_csv(fold_summary_path, index=False)
    margin_summary_df.to_csv(margin_summary_path, index=False)

    print(f"Saved fold summary to: {fold_summary_path}")
    print(f"Saved margin summary to: {margin_summary_path}")
    return fold_summary_df, margin_summary_df
