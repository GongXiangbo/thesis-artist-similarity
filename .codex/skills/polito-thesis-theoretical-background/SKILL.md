---
name: polito-thesis-theoretical-background
description: "Draft, revise, or critique the Theoretical Background chapter for this project-specific Politecnico di Torino English master's thesis on artist similarity, MIR, AI, machine learning, deep learning, embeddings, multimodal representation, Siamese networks, losses, metrics, and signal or text/image processing concepts. Use for definitions, equations, technical explanations, and prerequisite theory."
---

# Theoretical Background

## Style Anchor

Write only in polished academic English. Use the supplied Politecnico di Torino theses only as style and tone models: define broad concepts before specialized ones, introduce each model family through its purpose, and connect every theoretical concept to the later methodology. Do not reuse their wording, claims, citations, figures, or examples unless the current project sources support them.

Use an explanatory engineering style. The chapter should teach the minimum theory needed to understand the thesis, not survey all possible literature.

## Chapter Shape

Organize from general foundations to project-specific tools. A typical order is:

1. Music Information Retrieval or artist similarity concepts, if not introduced elsewhere.
2. Artificial intelligence and machine learning fundamentals.
3. Representation learning and embeddings.
4. Deep learning architectures relevant to the project, such as CNNs, transformers, or Siamese networks.
5. Modality-specific processing, such as audio features, image embeddings, text embeddings, or multimodal fusion.
6. Similarity measures, loss functions, dimensionality reduction, and evaluation concepts needed later.

Only include a subsection if it supports a method, metric, experiment, or interpretation used in the thesis.

## Explanation Pattern

For each concept:

1. Define it in one or two precise sentences.
2. Explain why it matters for this thesis.
3. Introduce the mathematical notation or architecture only as deeply as needed.
4. Cite foundational or authoritative sources.
5. End by linking the concept to the methodology chapter.

When using equations, define every symbol immediately after the equation and describe the intuition in prose. Do not leave equations as decorative material.

## Tone and Detail

Use present tense for general theory. Use neutral phrasing such as "is commonly used", "allows", "is designed to", and "can be interpreted as". Avoid conversational language and unqualified claims like "the best model".

Avoid:

- Long textbook digressions unrelated to the implemented method.
- Repeating implementation details from Methodology.
- Introducing models or metrics that never appear later.
- Overusing direct quotations for definitions; paraphrase and cite instead.

## Revision Checklist

Verify that each subsection has a visible reason to exist. Check that terms such as embedding, triplet loss, cosine similarity, fusion, latent space, precision, recall, and t-SNE are defined before they are used analytically in later chapters.
