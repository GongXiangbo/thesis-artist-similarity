---
name: polito-thesis-methodology
description: "Draft, revise, or critique the Methodology chapter for this project-specific Politecnico di Torino English master's thesis on artist similarity, problem formulation, data processing, multimodal embeddings, fusion, model architecture, training objective, triplet construction, similarity metrics, implementation details, and reproducibility."
---

# Methodology

## Style Anchor

Write only in polished academic English. Use the supplied Politecnico di Torino theses only as style and tone models: begin with a chapter roadmap, state the problem formally, then explain data processing, feature extraction, fusion, training, and representation learning in a reproducible order. Do not reuse their wording, claims, diagrams, citations, numbers, or experiments unless the current project sources support them.

Use precise engineering prose. The reader should be able to reimplement the method from this chapter together with the code repository and experiment chapter.

## Chapter Shape

Use this default progression:

1. Chapter overview.
2. Problem formulation and objective.
3. Dataset sources and inclusion criteria at the methodological level.
4. Preprocessing for each modality.
5. Embedding or feature extraction.
6. Feature normalization and fusion.
7. Artist-level vector construction.
8. Triplet or pair construction, if applicable.
9. Model architecture.
10. Loss function, distance metric, and training logic.
11. Implementation details needed for reproducibility.
12. Output representation and how similarity is computed.

Move full numerical results, comparison tables, and performance analysis to Experiments and Results.

## Writing Pattern

For each component, write in this order:

1. Purpose: explain why the component is needed.
2. Input: define the data or representation it receives.
3. Operation: describe the processing step, model, or equation.
4. Output: define the resulting representation and its dimensionality if known.
5. Rationale: justify the design choice briefly.

Use variables consistently and define them before equations. If a preprocessing or training choice is empirical, state that it is evaluated in the experiments chapter rather than presenting it as theoretically guaranteed.

## Tone and Tense

Use present tense for the proposed methodology: "the model receives", "the embeddings are concatenated", "the loss encourages". Use past tense only for design decisions already completed during experimentation.

Prefer:

- "The problem can be formulated as..."
- "The proposed architecture is composed of..."
- "Each artist is represented by..."
- "The distance between representations is computed as..."

Avoid:

- Claiming performance before showing results.
- Hiding assumptions about missing data, filtering, scaling, or label construction.
- Mixing code-level variable names with prose unless they are necessary for reproducibility.

## Revision Checklist

Check that every methodological choice has a reason, every equation has defined symbols, and every produced artifact is later evaluated. Ensure dataset numbers, embedding dimensions, hyperparameters, and model names match the code and experiment logs.
