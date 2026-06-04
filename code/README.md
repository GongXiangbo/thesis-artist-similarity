# Thesis Code - TripletNet1 hierarchical video version

This package keeps the original triplet-learning pipeline, but changes `TripletNet1` to match the current input format: each artist has up to 10 videos, each video has 30 uniformly sampled CLIP frame embeddings.

## Main changes

1. **`TripletNet1` now uses a hierarchical video-to-artist architecture**
   - Input: `(batch, videos<=10, frames=30, dim=768)`.
   - Backward compatible input: `(batch, frames, dim)` is treated as one valid video.
   - Video padding is inferred from all-zero video tensors.
   - Frame-level encoder has three branches only:
     - set/style attentive-statistics branch;
     - temporal Transformer branch with frame positional encoding;
     - first/second-order CLIP-delta branch.
   - Artist-level encoder has two masked branches:
     - masked video-set attentive-statistics branch;
     - masked self-attention branch over video tokens, without video positional encoding.
   - Final output uses BNNeck + L2 normalization and a learnable residual gate against the raw CLIP mean.

2. **`dataset.py` supports stacked artist tensors**
   - `process_artists(..., aggregation="stack", max_videos=10, num_frames=30)` returns `(10, 30, 768)` per artist.
   - Artists with fewer than 10 videos are zero-padded, not cyclically repeated.
   - `aggregation="mean"` is still available for legacy `TripletNet2`-`TripletNet5` comparisons.

3. **`experiment.py` defaults to the new TripletNet1 path**
   - Default `--artist-aggregation stack`.
   - Default `--videos-per-artist 10`.
   - Default `--num-frames 30`.
   - Default `--video-dropout 0.15`.

4. **Video dropout**
   - During training, valid videos are randomly dropped inside TripletNet1.
   - The mask always keeps at least one valid video per artist, so the model never receives an empty artist representation.

5. **Existing training improvements are preserved**
   - Artist-disjoint train/validation split.
   - Memory-bank semi-hard negative mining.
   - Known-positive and two-hop-neighbor exclusion for negative mining.

## Recommended command

```bash
python code/experiment.py \
  --base-dir data/video_embeddings \
  --triplets-csv data/triplets/triplets_ids_music_spot.csv \
  --model TripletNet1 \
  --artist-aggregation stack \
  --videos-per-artist 10 \
  --num-frames 30 \
  --video-dropout 0.15 \
  --margin 0.1 \
  --negative-mining memory_bank_semihard \
  --epochs 30
```

For legacy CNN baselines, use mean aggregation:

```bash
python code/experiment.py \
  --model TripletNet2 \
  --artist-aggregation mean \
  --margin 0.1 \
  --epochs 30
```

## Quick verification

```bash
cd code
pytest -q
```

The included tests check stacked-video input, zero-padding masks, backward-compatible single-video input, video dropout, and legacy mining utilities.
