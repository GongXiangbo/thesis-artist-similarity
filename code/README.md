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
   - Replaced `TripletNet1` with a four-branch CLIP-frame retrieval aggregator: frame-level L2 normalisation, 768→512 projection, Set/Style attention branch, multi-scale temporal convolution branch, 2-layer Temporal Transformer branch, first/second-order temporal-delta branch, CLIP mean residual anchoring, gated weighted-sum plus concatenation fusion, and 1024→512→256 projection head.

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

## Recommended negative mining

Use `--negative-mining memory_bank_semihard` for final TripletNet1 experiments. This implements global memory-bank hard-negative mining and avoids likely false negatives by masking known positives plus two-hop neighbours in the similarity graph. `batch_semihard` is still available for faster debugging, but it only mines within the current mini-batch.

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

## Important note about existing notebook results

`TripletNet1` has been replaced with the new four-branch CLIP-frame retrieval architecture.
The old saved outputs in `model1.ipynb` correspond to the previous TripletNet1 implementation and should be treated only as historical experiment records.
After this change, rerun the margin grid and compare models again before reporting final thesis results.

---

## Example command

```bash
python code/experiment.py \
  --base-dir data/video_embeddings \
  --triplets-csv data/triplets/triplets_ids_music_spot.csv \
  --model TripletNet1 \
  --margin 0.1 \
  --negative-mining memory_bank_semihard \
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
