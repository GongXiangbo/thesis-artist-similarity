from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from dataset import (
    build_negative_exclusion_map,
    create_artist_memory_bank,
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
    negative_mining: NegativeMiningMode = "memory_bank_semihard"
    mining_fallback: str = "closest_valid"
    grad_clip: float | None = 1.0


def _normalise_triplet_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(TRIPLET_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required triplet column(s): {sorted(missing)}")
    out = df.loc[:, list(TRIPLET_COLUMNS)].copy()
    for col in TRIPLET_COLUMNS:
        out[col] = out[col].astype(str)
    return out.drop_duplicates().reset_index(drop=True)


def _all_triplet_artists(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    return set(df[list(TRIPLET_COLUMNS)].to_numpy().reshape(-1))


def _rows_fully_inside_artists(df: pd.DataFrame, artists: set[str]) -> pd.Series:
    """Return rows where anchor, positive and negative are all inside artists."""
    return df[list(TRIPLET_COLUMNS)].isin(artists).all(axis=1)


def _build_artist_disjoint_fold_stats(
    *,
    fold_id: int,
    strategy: str,
    original_rows: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_artists_partition: set[str],
    val_artists_partition: set[str],
) -> dict[str, Any]:
    train_artist_set = _all_triplet_artists(train_df)
    val_artist_set = _all_triplet_artists(val_df)
    used_rows = len(train_df) + len(val_df)
    return {
        "fold": fold_id,
        "strategy": strategy,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "dropped_cross_partition_rows": original_rows - used_rows,
        "kept_row_ratio": used_rows / original_rows if original_rows else 0.0,
        "train_artists": len(train_artist_set),
        "val_artists": len(val_artist_set),
        "train_artist_partition_size": len(train_artists_partition),
        "val_artist_partition_size": len(val_artists_partition),
        "anchor_overlap": bool(set(train_df["anchor"]) & set(val_df["anchor"])),
        "artist_overlap": bool(train_artist_set & val_artist_set),
    }


def make_artist_disjoint_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 3407,
    *,
    shuffle_trials: int = 25,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    """Create K-fold CV splits with zero artist leakage.

    A row is assigned to a fold only when all three artists in that triplet
    (anchor, positive and negative) belong to the same side of the split. Rows
    crossing the train/validation artist boundary are dropped for that fold.

    This is stricter than anchor-group CV: no artist can appear anywhere in both
    train and validation, regardless of whether it appears as anchor, positive,
    or negative. If a fold cannot produce both train and validation triplets, the
    function raises an error instead of silently falling back to a leaky split.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if shuffle_trials < 1:
        raise ValueError("shuffle_trials must be at least 1")

    df = _normalise_triplet_df(df)
    if len(df) < n_splits:
        raise ValueError(f"Need at least {n_splits} triplets, found {len(df)}")

    artists = np.array(sorted(_all_triplet_artists(df)), dtype=object)
    if len(artists) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique artists for artist-disjoint {n_splits}-fold CV, "
            f"but only found {len(artists)}."
        )

    rng = np.random.default_rng(seed)
    best_folds: list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, Any]]] | None = None
    best_score: tuple[int, int, float] | None = None

    for _ in range(shuffle_trials):
        shuffled_artists = artists.copy()
        rng.shuffle(shuffled_artists)
        artist_folds = [set(map(str, fold.tolist())) for fold in np.array_split(shuffled_artists, n_splits)]

        candidate_folds = []
        min_val_rows = 10**18
        total_used_rows = 0
        total_val_rows = 0
        valid = True

        for fold_id, val_artist_partition in enumerate(artist_folds, start=1):
            train_artist_partition = set(map(str, shuffled_artists.tolist())) - val_artist_partition
            train_mask = _rows_fully_inside_artists(df, train_artist_partition)
            val_mask = _rows_fully_inside_artists(df, val_artist_partition)
            train_df = df.loc[train_mask].reset_index(drop=True)
            val_df = df.loc[val_mask].reset_index(drop=True)

            if train_df.empty or val_df.empty:
                valid = False
                break

            stats = _build_artist_disjoint_fold_stats(
                fold_id=fold_id,
                strategy="artist_disjoint_kfold",
                original_rows=len(df),
                train_df=train_df,
                val_df=val_df,
                train_artists_partition=train_artist_partition,
                val_artists_partition=val_artist_partition,
            )
            if stats["artist_overlap"]:
                raise RuntimeError(f"Internal error: fold {fold_id} is not artist-disjoint")

            candidate_folds.append((fold_id, train_df, val_df, stats))
            min_val_rows = min(min_val_rows, len(val_df))
            total_val_rows += len(val_df)
            total_used_rows += len(train_df) + len(val_df)

        if not valid:
            continue

        # Prioritize a usable validation set in every fold, then total retained
        # triplets, then average validation size.
        score = (min_val_rows, total_used_rows, total_val_rows / n_splits)
        if best_score is None or score > best_score:
            best_score = score
            best_folds = candidate_folds

    if best_folds is None:
        raise ValueError(
            "Could not create non-empty artist-disjoint folds. The triplet graph is too sparse for this n_splits. "
            "Try reducing n_splits, increasing the dataset size, or using a hold-out artist-disjoint split."
        )

    return best_folds


def make_anchor_group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 3407,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    """Deprecated compatibility wrapper.

    Older notebooks imported this name. It now returns artist-disjoint folds to
    prevent leakage through positive/negative artists. Use
    ``make_artist_disjoint_kfold_splits`` in new code for clarity.
    """
    return make_artist_disjoint_kfold_splits(df, n_splits=n_splits, seed=seed)


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
    *,
    negative_source_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    uses_dynamic_mining = config.negative_mining in {"batch_semihard", "memory_bank_semihard"}
    if uses_dynamic_mining:
        train_triplets = create_triplets_with_ids(train_df, artist_averages)
        # Priority 6: exclude direct known positives and two-hop neighbours from
        # the negative candidate pool. Passing the full filtered triplet graph as
        # negative_source_df also protects direct positives that were dropped by
        # artist-disjoint fold partitioning.
        positive_map = build_negative_exclusion_map(negative_source_df if negative_source_df is not None else train_df, symmetric=True, include_two_hop=True)
    else:
        train_triplets = create_triplets(train_df, artist_averages)
        positive_map = None
    val_triplets = create_triplets(val_df, artist_averages)
    if config.negative_mining == "memory_bank_semihard":
        memory_bank_ids, memory_bank_tensors = create_artist_memory_bank(train_df, artist_averages)
    else:
        memory_bank_ids, memory_bank_tensors = None, None

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
            memory_bank_ids=memory_bank_ids,
            memory_bank_tensors=memory_bank_tensors,
            memory_bank_batch_size=config.batch_size,
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
            "memory_bank_size": train_metrics.get("memory_bank_size", 0),
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
        if config.negative_mining in {"batch_semihard", "memory_bank_semihard"}:
            bank_bits = f" | bank={train_metrics.get('memory_bank_size', 0)}" if config.negative_mining == "memory_bank_semihard" else ""
            mining_bits = (
                f" | semi={train_metrics['semi_hard_ratio']:.1%}"
                f" | fallback={train_metrics['fallback_ratio']:.1%}"
                f" | skipped={train_metrics['skipped_ratio']:.1%}"
                f"{bank_bits}"
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
                "mean_memory_bank_size": _history_mean(item, "memory_bank_size"),
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
            mean_memory_bank_size=("mean_memory_bank_size", "mean"),
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
