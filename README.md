# Master Thesis Workspace

This workspace is organized by responsibility:

- `thesis/` - written thesis materials: drafts, figures, references, notes, and exports.
- `code/` - Python source files, notebooks, requirements, and experiment scripts.
- `data/` - research inputs, metadata, triplet definitions, and precomputed video embeddings.
- `results/` - trained models, checkpoints, experiment summaries, and archived cache files.

The main training entrypoint is:

```bash
python code/experiment.py --model TripletNet1 --margin 0.5 --epochs 30
```

By default it reads embeddings from `data/video_embeddings`, triplets from
`data/triplets/triplets_ids_spot.csv`, and writes checkpoints to
`results/checkpoints`.

## Leakage-safe evaluation

The training utilities now use **artist-disjoint** splitting by default. This means an artist cannot appear in both training and validation in any role: anchor, positive, or negative. Cross-boundary triplets are dropped rather than assigned to a split, so validation scores are no longer inflated by artist leakage.

For 5-fold experiments, the notebooks call `make_artist_disjoint_kfold_splits(...)`. The old `make_anchor_group_kfold_splits(...)` name is kept only as a compatibility wrapper and now also returns artist-disjoint folds.
