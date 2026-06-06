---
name: polito-thesis-experiments-results
description: "Draft, revise, or critique the Experiments and Results chapter for this project-specific Politecnico di Torino English master's thesis on artist similarity experiments, datasets, baselines, hyperparameters, ablations, evaluation metrics, tables, plots, t-SNE visualizations, result interpretation, limitations, and reproducibility."
---

# Experiments and Results

## Style Anchor

Write only in polished academic English. Use the supplied Politecnico di Torino theses only as style and tone models: open with a roadmap, describe the dataset and experimental setup before presenting results, keep metrics explicit, and interpret strengths and limitations without exaggeration. Do not reuse their wording, claims, tables, numbers, citations, or figures unless the current project sources support them.

Use empirical, evidence-led prose. Separate what was done, what was measured, what was observed, and what it means.

## Chapter Shape

Use this default progression:

1. Chapter overview.
2. Dataset description and filtering.
3. Experimental setup, including splits, hardware/software if relevant, optimizer, loss, batch size, learning rate, epochs, and random seeds when available.
4. Baselines and model variants.
5. Evaluation metrics, with definitions and interpretation.
6. Hyperparameter tuning or ablation studies.
7. Main quantitative results.
8. Qualitative analysis, retrieval examples, embedding inspection, or t-SNE plots.
9. Error analysis and limitations.
10. Short conclusive remarks connecting results to the thesis objective.

## Results Writing Pattern

For each experiment:

1. State the purpose of the experiment.
2. Describe the exact configuration.
3. Present the result in a table, figure, or concise paragraph.
4. Compare against a baseline or alternative.
5. Explain the interpretation.
6. State uncertainty, limitation, or possible cause when appropriate.

Avoid presenting a table without prose. The text should guide the reader to the main trend, not repeat every value.

## Metrics and Figures

Define metrics before using them analytically. For similarity and retrieval tasks, clarify whether higher or lower values are better and what constitutes a correct prediction. For visualizations such as t-SNE, state that they are qualitative aids and not definitive proof of cluster quality.

Caption tables and figures so they are understandable without reading the full paragraph. Use consistent terminology for model variants, modalities, datasets, and metrics.

## Tone and Tense

Use past tense for conducted experiments and present tense for interpreting tables or figures: "The model was trained...", "Table 5.2 reports...", "The results suggest...".

Avoid:

- "Clearly proves" or "perfectly captures".
- Claiming generalization beyond the tested dataset.
- Reporting high accuracy without discussing class balance, labels, or evaluation protocol.
- Moving methodological explanations that belong in Methodology into this chapter unless needed to understand an experiment.

## Revision Checklist

Verify all numbers against source logs, notebooks, or code outputs. Check that every model named in tables is defined, every metric is explained, and every major conclusion follows from a reported result.
