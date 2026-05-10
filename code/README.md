# Thesis Code - Organized and Optimized Version

## What was improved

This version keeps your original research direction and model definitions, but reorganizes the code so it is easier to read, reuse, and defend in a thesis.

### Main improvements
1. **Kept notebook compatibility**
   - `TripletNet1` to `TripletNet5` still exist.
   - `train.py`, `evaluate.py`, `dataset.py`, and `metrics.py` keep the same main API.

2. **Refactored `model.py`**
   - Added a shared `BaseTripletNet`.
   - Replaced repeated forward logic with reusable base classes.
   - Added a model registry and `build_model(...)`.
   - Removed hard-coded fully connected input sizes by inferring them automatically.

3. **Refactored `dataset.py`**
   - Removed CLIP loading at import time.
   - Added lazy CLIP loading for faster startup and cleaner training code.
   - Added reusable helpers:
     - `process_artists`
     - `filter_triplets`
     - `create_triplets`
     - `create_dataloaders`
   - Enforced artist-disjoint train/validation splitting by default to prevent leakage through anchor, positive, or negative artists.

4. **Improved engineering quality**
   - Added deterministic seed utility in `utils.py`.
   - Improved typing, comments, and error messages.
   - Reduced duplicated logic in training and evaluation.

5. **Added unified experiment script**
   - `experiment.py` can train any of the five models from one command.
   - This is cleaner than maintaining five nearly identical notebooks.

## Leakage-safe split protocol

Final experiments should use the strict artist-disjoint protocol now implemented in `dataset.py` and `cv_training.py`. In each fold or hold-out split, an artist is allowed to appear on only one side of the split, regardless of whether it appears as an anchor, positive, or negative artist. Triplets that cross the artist partition are dropped.

The expected split stats should report:

```text
artist_overlap: False
anchor_overlap: False
strategy: artist_disjoint or artist_disjoint_kfold
```

The previous anchor-only protocol is not recommended for final reporting because positive/negative artists can leak into validation.

---

## Recommended project structure

- `code/` - source code, notebooks, and reproducible experiment entrypoints
- `data/metadata/` - artist metadata and related-artist CSV files
- `data/triplets/` - triplet CSV files used for training/evaluation
- `data/video_embeddings/` - artist folders with precomputed video embeddings
- `results/checkpoints/` - trained checkpoint files and checkpoint histories
- `results/models/` - saved model weight exports
- `results/summaries/` - compact experiment summaries
- `results/cache/` - archived Python and Jupyter cache files
- `thesis/` - drafts, figures, references, notes, and exports for the written thesis

---

## Which model looked best from your existing notebooks

Based on the saved notebook outputs in this project:

- **Best overall:** `TripletNet1`
- **Best recorded validation accuracy:** **98.13%**
- **Best margin in the saved notebook output:** **0.5**

Other strong results:
- `TripletNet4`: 97.56%
- `TripletNet5`: 97.56%
- `TripletNet3`: 97.40%
- `TripletNet2`: 96.83%

So if your thesis needs **one best model**, I recommend using **TripletNet1** as the primary model and presenting the others as comparative baselines.

---

## Example command

```bash
python code/experiment.py \
  --base-dir data/video_embeddings \
  --triplets-csv data/triplets/triplets_ids_spot.csv \
  --model TripletNet1 \
  --margin 0.5 \
  --negative-mining batch_semihard \
  --epochs 30
```

---

## Suggested thesis presentation angle

You can describe the engineering upgrade like this:

> The experimental pipeline was refactored into reusable modules for model definition, data preparation, training, evaluation, and experiment control.  
> This reduced duplication across notebooks, improved reproducibility, and preserved backward compatibility with the original experimental setup.

---

## Notes

- The original notebooks are still useful as experiment records.
- For future work, prefer `experiment.py` or `model_benchmark.ipynb` instead of copying notebook code for each model.
