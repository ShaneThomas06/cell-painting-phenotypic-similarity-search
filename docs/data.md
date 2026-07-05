# Data Notes

## Primary Dataset

This project starts with RxRx1, Recursion's public Cell Painting benchmark for studying experimental batch effects in biological image screens.

Official sources:

- Dataset page: https://rxrx.ai/rxrx1
- Kaggle competition page: https://www.kaggle.com/c/recursion-cellular-image-classification
- Paper: https://arxiv.org/abs/2301.05768

## Why RxRx1 Fits This Project

RxRx1 was designed around the same problem this project studies: distinguishing biological perturbation signal from technical batch variation.

Important properties for the project:

- Images are six-channel Cell Painting microscopy images.
- Each image represents one imaging site from one well.
- Each experiment corresponds to a batch-like context.
- The biological label is siRNA perturbation identity.
- Held-out experiment evaluation is central to the benchmark.

## Initial Scope

The first analysis should not use the full dataset immediately. Start with:

- one cell type
- a subset of perturbations with enough images across batches
- a held-out batch split
- 224 x 224 image inputs for the baseline model

## Expected Metadata Columns

The starter code expects a normalized metadata table with at least these columns:

```text
image_path
perturbation
batch
cell_type
```

The `image_path` value should be relative to the configured image root.

## First Data Milestone

Before model training, run metadata inspection and answer:

1. How many rows are available?
2. Which columns are present?
3. How many perturbations are present?
4. How many batches are present?
5. How many images exist per cell type?
6. Which cell type is best for the first subset?
7. Which batches should be held out for testing?

## Biological Caution

RxRx1 uses siRNA perturbations, and siRNAs can have off-target effects. Describe labels as perturbation or siRNA identity, not as definitive gene-function effects.
