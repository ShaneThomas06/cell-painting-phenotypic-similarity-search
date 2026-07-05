# Batch-Robust Cell Painting Embeddings for Perturbation Profiling

This project learns morphology-based image embeddings from Cell Painting microscopy images and evaluates whether those embeddings recover biological perturbation signal across experimental batches.

The initial focus is the RxRx1 dataset, a high-content fluorescence microscopy benchmark designed for studying batch effects in biological image screens.

## Research Question

Can deep learning embeddings recover genetic perturbation identity from Cell Painting images across experimental batches?

## Motivation

Cell Painting is used in phenotypic screening and drug discovery to measure how cells respond to genetic or chemical perturbations. A central challenge is that high-content imaging data can contain strong batch effects: technical variation from plates, sites, cell types, imaging runs, or protocols may obscure the biological signal.

This project treats batch robustness as the core scientific problem. The goal is not only to train an image classifier, but to evaluate whether learned image representations preserve perturbation-level morphology while generalizing across held-out batches.

## Dataset

Primary dataset:

- RxRx1

Initial scope:

- One cell type
- A balanced subset of perturbations
- Multiple experimental batches
- Held-out batch evaluation
- Multi-channel fluorescence microscopy images

Large raw image files should not be committed to the repository. The `data/` directory contains placeholders only.

See [data notes](docs/data.md) for dataset assumptions and the first metadata-inspection milestone.

## Planned Method

1. Inspect RxRx1 metadata and image structure.
2. Build a balanced subset across perturbations and batches.
3. Train a baseline image encoder to predict perturbation identity.
4. Extract penultimate-layer embeddings.
5. Evaluate cross-batch perturbation retrieval.
6. Visualize embeddings colored by perturbation and batch.

## Initial Baseline

- Encoder: ResNet18
- Objective: perturbation classification
- Split strategy: batch-aware train/validation/test split
- Main embedding evaluation: nearest-neighbor retrieval across held-out batches

## Evaluation

Primary metrics:

- Top-1 and top-5 perturbation accuracy
- Balanced accuracy
- Macro F1
- Cross-batch nearest-neighbor retrieval accuracy
- Mean average precision for perturbation retrieval

Representation diagnostics:

- UMAP colored by perturbation
- UMAP colored by batch
- Confusion matrix
- Retrieval image panels
- Batch distribution plots

## Setup

See [setup instructions](docs/setup.md).

## Repository Structure

```text
configs/                         Experiment and subset configuration files
data/                            Local data folders; raw data is ignored by git
notebooks/                       Exploration and result-analysis notebooks
reports/figures/                 Generated plots for README and reports
reports/tables/                  Generated metrics tables
src/cell_painting_profiling/     Reusable Python package
tests/                           Lightweight tests for data and transforms
```

## Milestones

- [ ] Project scaffold
- [ ] RxRx1 metadata inspection
- [ ] Balanced subset definition
- [ ] Dataset loader
- [ ] Baseline ResNet18 training
- [ ] Embedding extraction
- [ ] Cross-batch retrieval evaluation
- [ ] UMAP and retrieval visualizations
- [ ] Final README results and interpretation

## References

- RxRx1: A Dataset for Evaluating Experimental Batch Correction Methods
- Cell Painting Gallery: an open resource for image-based profiling
- MorphoHELM: A Comprehensive Benchmark for Evaluating Representations for Microscopy-Based Morphology Assays


