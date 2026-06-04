# Master Thesis Workspace

This workspace is organized by responsibility:

- `code/` - Python source files, notebooks, requirements, and experiment scripts.
- `data/` - metadata, triplet definitions, and precomputed video embeddings.
- `results/` - trained models, checkpoints, experiment summaries, and caches.

## Current TripletNet1 setup

`TripletNet1` has been changed for the current video-only input format:

```text
artist -> <=10 videos -> 30 CLIP frame embeddings per video
```

The default training path now keeps the video structure instead of averaging all videos first. Artists with fewer than 10 videos are zero-padded, and the model infers a valid-video mask from the zero pads.

```bash
python code/experiment.py \
  --model TripletNet1 \
  --artist-aggregation stack \
  --videos-per-artist 10 \
  --num-frames 30 \
  --video-dropout 0.15 \
  --margin 0.1 \
  --epochs 30
```

By default it reads embeddings from `data/video_embeddings`, triplets from `data/triplets/triplets_ids_music_spot.csv`, and writes checkpoints to `results/checkpoints`.

## What changed in TripletNet1

- frame-to-video set/style branch;
- frame-to-video temporal Transformer branch;
- frame-to-video CLIP-delta branch;
- dimension-wise branch fusion;
- artist-level masked set/context encoder over video tokens;
- BNNeck + L2-normalized output;
- sample-dependent raw CLIP residual gate;
- training-time video dropout that always keeps at least one valid video.

## Legacy baselines

`TripletNet2`-`TripletNet5` still expect legacy `(frames, dim)` artist tensors. Use mean aggregation for those models:

```bash
python code/experiment.py --model TripletNet2 --artist-aggregation mean --margin 0.1 --epochs 30
```

## Hard-negative mining and safer negative sampling

The recommended training mode is `memory_bank_semihard`. At the start of each epoch, the current model encodes every training artist into a memory bank. For each anchor-positive pair, the miner searches the full training-artist pool for a semi-hard or closest valid negative, instead of being limited to artists inside the current mini-batch.

To reduce false negatives, the negative candidate mask excludes the anchor itself, direct known positives from the similarity graph, and two-hop neighbours in the similarity graph.

## Leakage-safe evaluation

The training utilities use artist-disjoint splitting by default. An artist cannot appear in both training and validation in any role: anchor, positive, or negative. Cross-boundary triplets are dropped rather than assigned to a split, so validation scores are not inflated by artist leakage.
