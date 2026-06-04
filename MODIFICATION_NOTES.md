# TripletNet1 Simple Transformer + Video Dropout Modification Notes

Implemented changes:

1. `TripletNet1` now supports hierarchical artist tensors shaped `(videos, frames, embedding_dim)`.
2. `dataset.process_artists(..., aggregation="stack")` keeps up to 10 videos per artist and zero-pads missing videos.
3. `TripletNet1` infers valid videos from all-zero padded video tensors.
4. Frame embeddings are projected to `model_dim` with Linear/LayerNorm.
5. Frame-to-video encoding uses a frame-level `TransformerEncoder` and mean pooling.
6. Artist-level encoding uses a masked `TransformerEncoder` over valid video tokens, followed by masked mean pooling.
7. Final embedding uses a projection head, BNNeck, and L2 normalization.
8. Training-time video dropout is enabled through `--video-dropout`, default `0.15`.
9. `experiment.py` defaults to `--artist-aggregation stack`, `--videos-per-artist 10`, and `--num-frames 30`.

Verification run:

```bash
cd code
OMP_NUM_THREADS=1 pytest -q
# 13 passed, 1 warning
```

## model1.ipynb update
- `model1.ipynb` now loads artist embeddings with `ARTIST_AGGREGATION = "stack"`, `MAX_VIDEOS = 10`, `NUM_FRAMES = 30`, and `VIDEO_DROPOUT = 0.15`.
- `MODEL_KWARGS` is inferred from `(max_videos, num_frames, d_model)` stack-shaped embeddings and passes `video_dropout_p` into `TripletNet1`.
- The margin grid in `model1.ipynb` is now `MARGINS = [0.1, 0.3, 0.5, 0.7, 0.9]`.
