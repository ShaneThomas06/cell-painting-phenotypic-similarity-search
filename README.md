# Phenotypic Similarity Search for Drug Mechanism Discovery with Cell Painting

This project uses Cell Painting microscopy images to learn deep phenotypic embeddings of chemical and genetic perturbations. The goal is to build a morphology-based retrieval workflow: given a query compound or perturbation, find other perturbations that produce similar cellular phenotypes and evaluate whether those visual neighbors share known mechanisms, targets, or pathway annotations.

The project is framed as a drug-discovery analysis workflow, not a leaderboard-style image classifier.

## Research Question

Can CNN-derived Cell Painting embeddings identify biologically meaningful perturbation similarity and recover known mechanism-of-action relationships?

## Motivation

Cell Painting captures rich cellular morphology across multiple fluorescent channels, measuring how perturbations affect nuclei, cytoplasm, mitochondria, actin, endoplasmic reticulum, Golgi, nucleoli, and plasma membrane-associated structures. In phenotypic drug discovery, compounds that produce similar morphology may share pathway activity, target biology, toxicity signals, or downstream cellular responses even when their chemical structures differ.

This project treats morphology as a searchable biological fingerprint. Instead of training a model only to classify labels, it asks whether image-derived embeddings can support mechanism discovery, compound prioritization, and biological interpretation.

## Core Idea

1. Load Cell Painting images and perturbation metadata.
2. Train or fine-tune a CNN image encoder.
3. Extract image-level embeddings from microscopy images.
4. Aggregate image-level embeddings into perturbation-level phenotypic fingerprints.
5. Build a nearest-neighbor search workflow for perturbations.
6. Compare phenotypic similarity against mechanism, target, pathway, and chemical metadata.
7. Highlight interpretable examples where morphology agrees or disagrees with known annotations.

## Unique Edge

The main analysis focuses on phenotypic retrieval:

```text
query perturbation -> closest morphology neighbors -> mechanism/target comparison
```

A secondary analysis will examine morphology-chemistry disagreement:

```text
chemically dissimilar but morphologically similar perturbations
```

These cases are especially interesting for drug discovery because they may suggest shared downstream biology, convergent cellular response, unexpected off-target effects, or repurposing hypotheses.

## Candidate Data Sources

Primary direction:

- Cell Painting Gallery
- JUMP Cell Painting Consortium data

Fallback or benchmarking direction:

- RxRx1 for batch-effect-aware representation testing

The first implementation will inspect available metadata before choosing the final subset. Raw image data should not be committed to the repository.

See [data notes](docs/data.md), [dataset reconnaissance](docs/dataset_reconnaissance.md), and the [baseline CNN plan](docs/baseline_cnn_plan.md) for dataset assumptions, candidate rankings, and baseline-training direction.

## Planned Method

1. Inspect Cell Painting metadata and identify compound, perturbation, mechanism, target, batch, plate, and image fields.
2. Build a manageable subset with enough replicate images per perturbation.
3. Train a baseline CNN encoder or use a pretrained encoder as a feature extractor.
4. Extract image embeddings and aggregate them to perturbation fingerprints.
5. Build nearest-neighbor retrieval over perturbation fingerprints.
6. Evaluate whether nearest neighbors share known mechanisms, targets, or pathways.
7. Visualize representative image panels, UMAP embeddings, retrieval examples, and disagreement cases.

## Initial Baseline

- Encoder: ResNet18 or EfficientNet-B0
- Input: multi-channel Cell Painting image composites or channel-aware tensors
- Unit of analysis: perturbation-level fingerprint
- Retrieval metric: cosine similarity between fingerprints
- Primary evaluation: top-k mechanism or target recovery among nearest neighbors

## Evaluation

Retrieval metrics:

- top-k mechanism recovery
- top-k target recovery
- mean average precision for shared mechanism
- enrichment of shared annotations among nearest neighbors

Representation diagnostics:

- UMAP colored by mechanism, target, batch, and perturbation type
- nearest-neighbor retrieval panels
- perturbation similarity heatmaps
- morphology-chemistry agreement and disagreement examples
- batch distribution checks

## Setup

See [setup instructions](docs/setup.md).

## Repository Structure

```text
configs/                         Experiment and subset configuration files
data/                            Local data folders; raw data is ignored by git
docs/                            Project design, setup, and data notes
notebooks/                       Exploration and result-analysis notebooks
reports/figures/                 Generated plots for README and reports
reports/tables/                  Generated metrics tables
src/cell_painting_profiling/     Reusable Python package
tests/                           Lightweight tests for data and analysis utilities
```

## Milestones

- [x] Project scaffold
- [x] Python environment and starter tests
- [x] Cell Painting metadata inspection
- [x] Dataset and subset selection
- [x] Image loading and preprocessing
- [x] Baseline CNN embedding extraction smoke test
- [x] MOA-balanced baseline training subset
- [x] Perturbation-level fingerprint aggregation smoke test
- [x] Phenotypic nearest-neighbor search smoke test
- [ ] Mechanism and target recovery analysis
- [ ] Morphology-chemistry disagreement analysis
- [ ] Final figures and interpretation

## References

- Cell Painting Gallery: an open resource for image-based profiling
- JUMP Cell Painting Consortium
- Cell Painting: a decade of discovery and innovation in cellular imaging
- Image-based profiling for drug discovery

