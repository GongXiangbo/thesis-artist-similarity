"""Unified training entrypoint for comparing TripletNet1~4.

Example:
    python code/experiment.py --model TripletNet1 --margin 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim

from dataset import (
    create_dataloaders_from_triplet_lists,
    create_triplets,
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
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam"])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--distance-fn", type=str, default="cosine", choices=["cosine", "euclidean"])
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
            "Use --artist-aggregation mean for TripletNet2-4."
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

    train_triplets = create_triplets(train_df, artist_averages)
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

    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer!r}")
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
        )
        val_metrics = evaluate(model, val_loader, criterion, device, args.distance_fn, return_details=True)
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch,
            "optimizer": args.optimizer,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["ranking_acc"],
            "train_triplet_acc": train_metrics["ranking_acc"],
            "train_margin_acc": train_metrics["margin_acc"],
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
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.5f} | train_acc={train_metrics['ranking_acc']:.2%} | "
            f"val_loss={val_metrics['loss']:.5f} | val_acc={val_metrics['ranking_acc']:.2%} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
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
