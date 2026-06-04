"""Unified training entrypoint for comparing TripletNet1~5.

Example:
    python code/experiment.py --model TripletNet1 --margin 0.5 --negative-mining memory_bank_semihard
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim

from dataset import (
    build_negative_exclusion_map,
    create_artist_memory_bank,
    create_dataloaders_from_triplet_lists,
    create_triplets,
    create_triplets_with_ids,
    filter_triplets,
    infer_artist_tensor_shape,
    infer_embedding_shape,
    process_artists,
    split_triplet_dataframe_by_artist,
)
from evaluate import evaluate
from metrics import cosine_distance
from model import build_model
from train import train
from utils import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="data/video_embeddings")
    parser.add_argument("--triplets-csv", type=str, default="data/triplets/triplets_ids_music_spot.csv")
    parser.add_argument("--model", type=str, default="TripletNet1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--distance-fn", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument(
        "--negative-mining",
        type=str,
        default="memory_bank_semihard",
        choices=["fixed", "random", "batch_semihard", "memory_bank_semihard"],
        help="'memory_bank_semihard' mines hard negatives from all training artists each epoch; validation stays fixed.",
    )
    parser.add_argument("--mining-fallback", type=str, default="closest_valid")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-dir", type=str, default="results/checkpoints")
    parser.add_argument(
        "--artist-aggregation",
        type=str,
        default="stack",
        choices=["mean", "stack"],
        help="Use 'stack' for hierarchical TripletNet1: (videos, frames, dim) with zero padding.",
    )
    parser.add_argument(
        "--videos-per-artist",
        type=int,
        default=10,
        help="Maximum number of videos retained per artist in stack mode.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Frame embeddings per video in stack mode; use the default 30 for current data.",
    )
    parser.add_argument(
        "--video-dropout",
        type=float,
        default=0.15,
        help="Training-time probability of dropping each valid video inside TripletNet1.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = resolve_project_path(args.base_dir)
    triplets_csv = resolve_project_path(args.triplets_csv)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artist_averages = process_artists(
        base_dir,
        aggregation=args.artist_aggregation,
        max_videos=args.videos_per_artist,
        num_frames=args.num_frames,
    )
    if not artist_averages:
        raise RuntimeError(f"No artist embeddings found under {base_dir}")
    seq_len, d_model = infer_embedding_shape(artist_averages)
    artist_tensor_shape = infer_artist_tensor_shape(artist_averages)
    inferred_videos = int(artist_tensor_shape[0]) if len(artist_tensor_shape) == 3 else 1
    if len(artist_tensor_shape) == 3 and args.model != "TripletNet1":
        raise RuntimeError(
            "aggregation='stack' produces 4D batches and is currently supported only by TripletNet1. "
            "Use --artist-aggregation mean for TripletNet2-5."
        )

    df = pd.read_csv(triplets_csv)
    filtered_df = filter_triplets(df, artist_averages)
    if filtered_df.empty:
        raise RuntimeError("No usable triplets after filtering. Check triplet CSV IDs and embedding folders.")

    train_df, val_df, split_stats = split_triplet_dataframe_by_artist(
        filtered_df,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(f"Split stats: {split_stats}")
    print(
        f"Inferred artist tensor shape: {artist_tensor_shape} | "
        f"frames={seq_len}, d_model={d_model}, videos={inferred_videos}"
    )

    uses_dynamic_mining = args.negative_mining in {"batch_semihard", "memory_bank_semihard"}
    if uses_dynamic_mining:
        train_triplets = create_triplets_with_ids(train_df, artist_averages)
        # Priority 6: exclude known positives and two-hop neighbours so that
        # hard negatives are less likely to be unlabelled true positives.
        positive_map = build_negative_exclusion_map(filtered_df, symmetric=True, include_two_hop=True)
    else:
        train_triplets = create_triplets(train_df, artist_averages)
        positive_map = None
    if args.negative_mining == "memory_bank_semihard":
        memory_bank_ids, memory_bank_tensors = create_artist_memory_bank(train_df, artist_averages)
    else:
        memory_bank_ids, memory_bank_tensors = None, None
    val_triplets = create_triplets(val_df, artist_averages)
    train_loader, val_loader = create_dataloaders_from_triplet_lists(
        train_triplets,
        val_triplets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model_kwargs = {"d_model": d_model, "seq_len": seq_len}
    if args.model == "TripletNet1":
        model_kwargs.update({"max_videos": inferred_videos, "video_dropout_p": args.video_dropout})
    model = build_model(args.model, **model_kwargs).to(device)
    if args.distance_fn == "cosine":
        criterion = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=args.margin, swap=True)
    else:
        criterion = nn.TripletMarginLoss(margin=args.margin, swap=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.distance_fn,
            return_details=True,
            negative_mining=args.negative_mining,
            positive_map=positive_map,
            mining_fallback=args.mining_fallback,
            memory_bank_ids=memory_bank_ids,
            memory_bank_tensors=memory_bank_tensors,
            memory_bank_batch_size=args.batch_size,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, args.distance_fn, return_details=True)
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch,
            "negative_mining": args.negative_mining,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["ranking_acc"],
            "train_triplet_acc": train_metrics["ranking_acc"],
            "train_margin_acc": train_metrics["margin_acc"],
            "semi_hard_ratio": train_metrics["semi_hard_ratio"],
            "fallback_ratio": train_metrics["fallback_ratio"],
            "skipped_ratio": train_metrics["skipped_ratio"],
            "mean_pos_dist": train_metrics["mean_pos_dist"],
            "mean_neg_dist": train_metrics["mean_neg_dist"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["ranking_acc"],
            "val_margin_acc": val_metrics["margin_acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if (val_metrics["ranking_acc"] > best_val_acc) or (
            val_metrics["ranking_acc"] == best_val_acc and val_metrics["loss"] < best_val_loss
        ):
            best_val_acc = val_metrics["ranking_acc"]
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        mining_bits = ""
        if args.negative_mining in {"batch_semihard", "memory_bank_semihard"}:
            bank_bits = f" | bank={train_metrics.get('memory_bank_size', 0)}" if args.negative_mining == "memory_bank_semihard" else ""
            mining_bits = (
                f" | semi={train_metrics['semi_hard_ratio']:.1%}"
                f" | fallback={train_metrics['fallback_ratio']:.1%}"
                f" | skipped={train_metrics['skipped_ratio']:.1%}"
                f"{bank_bits}"
            )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.5f} | train_acc={train_metrics['ranking_acc']:.2%} | "
            f"val_loss={val_metrics['loss']:.5f} | val_acc={val_metrics['ranking_acc']:.2%} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}{mining_bits}"
        )

    history_path = output_dir / f"{args.model}_margin_{args.margin}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    if best_state is not None:
        checkpoint_path = output_dir / f"{args.model}_margin_{args.margin}_best.pt"
        torch.save(best_state, checkpoint_path)
        print(f"Saved best checkpoint to {checkpoint_path}")
    print(f"Best validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
