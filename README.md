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
