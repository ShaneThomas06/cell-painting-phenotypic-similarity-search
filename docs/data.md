# Data Notes

## Primary Data Direction

This project should prioritize Cell Painting datasets with both microscopy images and perturbation annotations that support mechanism or target interpretation.

Preferred sources:

- Cell Painting Gallery
- JUMP Cell Painting Consortium data

Fallback or benchmark source:

- RxRx1, useful for batch-effect-aware representation testing but not ideal as the central project because it is closely tied to a known classification benchmark.

## Why Metadata Matters

The project is a phenotypic similarity search workflow. Images are used to learn morphology, but metadata is needed to interpret whether visually similar perturbations share biological meaning.

Useful metadata fields include:

```text
image_path
compound_id
compound_name
perturbation_id
perturbation_type
mechanism_of_action
target
pathway
smiles
batch
plate
well
site
cell_line
channel
```

Not every dataset will contain every field. The first data milestone is to inspect available metadata and decide which evaluation questions are feasible.

## Initial Dataset Selection Criteria

A strong first subset should have:

- enough replicate images per perturbation
- multiple perturbations per mechanism or target class
- enough metadata to compare visual similarity with known biology
- manageable image volume for local development
- batch, plate, or experiment labels for technical-variation checks

## Project-Level Unit Of Analysis

The model may process individual images, but the main scientific object is the perturbation-level fingerprint.

```text
image embeddings -> replicate aggregation -> perturbation fingerprint -> nearest-neighbor search
```

This avoids treating single fields of view as independent biological conclusions.

## First Data Milestone

Before model training, metadata inspection should answer:

1. How many image records are available?
2. Which image path or channel columns are present?
3. Which perturbation identifiers are available?
4. How many perturbations have mechanism, target, or pathway labels?
5. How many replicate images exist per perturbation?
6. Are batch, plate, well, site, or cell-line fields available?
7. Which subset supports the strongest retrieval evaluation?

## Evaluation-Ready Metadata

For the first retrieval benchmark, the ideal normalized metadata table should include at least:

```text
image_path
perturbation_id
batch
```

For mechanism discovery analysis, it should also include one or more of:

```text
mechanism_of_action
target
pathway
compound_name
smiles
```

## Biological Caution

Morphological similarity is evidence of related cellular response, not proof of shared direct target. Final interpretations should distinguish between direct mechanism, downstream pathway convergence, toxicity response, and technical artifact.
