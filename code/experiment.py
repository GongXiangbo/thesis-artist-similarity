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
    infer_embedding_shape,
    process_artists,
    split_triplet_dataframe_by_artist,
)
from evaluate import evaluate
from metrics import cosine_distance
from model import build_model
from train import train
from utils import configure_torch_runtime, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="data/video_embeddings")
    parser.add_argument("--triplets-csv", type=str, default="data/triplets/triplets_ids_spot.csv")
    parser.add_argument("--model", type=str, default="TripletNet1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers. Use -1 to auto-select a CUDA-friendly value.",
    )
    parser.add_argument("--memory-bank-batch-size", type=int, default=0)
    parser.add_argument(
        "--memory-bank-device-cache",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Cache memory-bank input tensors on the GPU when possible.",
    )
    parser.add_argument("--amp", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--matmul-precision", type=str, default="high", choices=["highest", "high", "medium"])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
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
    return parser.parse_args()


def _resolve_amp_enabled(value: str, device: torch.device | str) -> bool:
    if value == "off":
        return False
    return torch.device(device).type == "cuda"


def _resolve_memory_bank_device_cache(value: str) -> bool | None:
    if value == "on":
        return True
    if value == "off":
        return False
    return None


def _state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    state_source = getattr(model, "_orig_mod", model)
    return {key: value.detach().cpu().clone() for key, value in state_source.state_dict().items()}


def _make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    configure_torch_runtime(
        deterministic=args.deterministic,
        matmul_precision=args.matmul_precision,
        allow_tf32=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = _resolve_amp_enabled(args.amp, device)
    memory_bank_device_cache = _resolve_memory_bank_device_cache(args.memory_bank_device_cache)
    base_dir = resolve_project_path(args.base_dir)
    triplets_csv = resolve_project_path(args.triplets_csv)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artist_averages = process_artists(base_dir)
    if not artist_averages:
        raise RuntimeError(f"No artist embeddings found under {base_dir}")
    seq_len, d_model = infer_embedding_shape(artist_averages)

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
    print(f"Inferred embedding shape: seq_len={seq_len}, d_model={d_model}")

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

    model = build_model(args.model, d_model=d_model, seq_len=seq_len).to(device)
    if args.compile_model:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
        else:
            print("torch.compile is not available in this PyTorch build; continuing without compilation.")
    if args.distance_fn == "cosine":
        criterion = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=args.margin, swap=True)
    else:
        criterion = nn.TripletMarginLoss(margin=args.margin, swap=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3)
    scaler = _make_grad_scaler(amp_enabled)

    print(
        "Runtime: "
        f"device={device} | amp={amp_enabled} ({args.amp_dtype}) | "
        f"batch_size={args.batch_size} | num_workers={args.num_workers} | "
        f"memory_bank_batch_size={args.memory_bank_batch_size or max(1024, args.batch_size * 4)} | "
        f"memory_bank_device_cache={args.memory_bank_device_cache}"
    )

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
            memory_bank_batch_size=args.memory_bank_batch_size or None,
            memory_bank_device_cache=memory_bank_device_cache,
            amp_enabled=amp_enabled,
            amp_dtype=args.amp_dtype,
            scaler=scaler,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.distance_fn,
            return_details=True,
            amp_enabled=amp_enabled,
            amp_dtype=args.amp_dtype,
        )
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
            "amp_enabled": train_metrics["amp_enabled"],
            "memory_bank_cached_on_device": train_metrics["memory_bank_cached_on_device"],
        }
        history.append(row)
        if (val_metrics["ranking_acc"] > best_val_acc) or (
            val_metrics["ranking_acc"] == best_val_acc and val_metrics["loss"] < best_val_loss
        ):
            best_val_acc = val_metrics["ranking_acc"]
            best_val_loss = val_metrics["loss"]
            best_state = _state_dict_to_cpu(model)
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
