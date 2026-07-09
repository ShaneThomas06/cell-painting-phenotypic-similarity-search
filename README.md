# Phenotypic Similarity Search for Drug Mechanism Discovery with Cell Painting

This project builds a Cell Painting-based phenotypic similarity search workflow for drug mechanism discovery. It uses multi-channel microscopy images to generate CNN-derived compound fingerprints, then evaluates whether visually similar compounds share mechanism-of-action annotations or chemical structure similarity.

## Research Question

Can Cell Painting image embeddings recover biologically meaningful compound similarity, and can morphology highlight relationships that are not captured by chemical structure alone?

## Why This Project

Cell Painting measures cellular morphology across multiple fluorescence channels. In drug discovery, compounds that produce similar cellular phenotypes may share pathway activity, downstream biological response, toxicity signatures, or mechanism-related behavior even when their chemical structures differ.

This project treats morphology as a searchable biological fingerprint. The main output is a compound-level retrieval workflow that connects microscopy images, mechanism labels, and chemical similarity.

## Completed Workflow

```text
metadata normalization
image download and validation
multi-channel image loading
ResNet18 representation learning
image-level embedding extraction
compound-level fingerprint aggregation
nearest-neighbor phenotypic retrieval
morphology-chemistry comparison
case-study visualization
```

## Dataset

The benchmark uses the `cpg0002-jump-scope` JUMP-MOA Cell Painting dataset.

Final benchmark subset:

```text
8 mechanism-of-action classes
2 compounds per mechanism
12 image sites per compound
192 image records
960 channel-level image files
```

Each image record contains five fluorescence channels:

```text
RNA, Mito, AGP, ER, DNA
```

## Headline Results

The strongest top-1 retrieval result came from the pretrained fine-tuned ResNet18 model. The strongest top-3 query coverage came from the frozen pretrained ResNet18 representation. Overall performance remains limited under compound-holdout validation, which indicates that the current bottleneck is representation quality and compound diversity rather than basic pipeline implementation.

![Mechanism-aware retrieval comparison](reports/figures/model_comparison_retrieval.png)

Compound-level UMAP projections show how representation choice changes the organization of morphology-derived fingerprints.

![Compound embedding UMAP](reports/figures/compound_embedding_umap.png)

Representative retrieval examples show one top-1 mechanism match and one case where the correct mechanism appears at rank 3 rather than rank 1.

![Retrieval examples](reports/figures/retrieval_example_panel.png)

## Morphology-Chemistry Analysis

The morphology-chemistry analysis compares CNN-derived phenotypic similarity with Morgan fingerprint chemical similarity. The strongest disagreement case is a CDC inhibitor pair, BMS-863233 and KH-CB19, with high morphology similarity and low chemical similarity. This supports the main project direction: Cell Painting can highlight shared cellular response that is not captured by chemical structure alone.

![Morphology and chemistry similarity](reports/figures/morphology_chemistry_similarity.png)

The case-study panel shows the paired Cell Painting composites behind the strongest agreement and disagreement examples.

![Morphology-chemistry case studies](reports/figures/morphology_chemistry_case_studies.png)

## Interpretation

The project supports a representation-centered conclusion. The full pipeline works, but the current dataset scale and generic pretrained image features limit mechanism recovery. The strongest contribution is the reproducible workflow: image-derived morphology fingerprints can be compared against mechanism labels and chemical structure similarity to identify interpretable agreement and disagreement cases.

See the [final report](docs/final_report.md) for the complete scientific narrative and [representation benchmark results](docs/results.md) for metric details.

## Setup

See [setup instructions](docs/setup.md).

## Repository Structure

```text
configs/                         Experiment and subset configuration files
data/                            Local data folders; raw data is ignored by git
docs/                            Project design, setup, and final reports
notebooks/                       Exploration and result-analysis notebooks
reports/figures/                 Generated plots for README and reports
reports/tables/                  Generated metrics tables
src/cell_painting_profiling/     Reusable Python package
tests/                           Lightweight tests for data and analysis utilities
```

## Main Artifacts

```text
docs/final_report.md
reports/tables/model_comparison_summary.csv
reports/tables/morphology_chemistry_pairwise.csv
reports/tables/morphology_chemistry_cases.csv
reports/figures/model_comparison_retrieval.png
reports/figures/retrieval_example_panel.png
reports/figures/morphology_chemistry_similarity.png
reports/figures/morphology_chemistry_case_studies.png
```

## Milestones

- [x] Project scaffold
- [x] Python environment and starter tests
- [x] Cell Painting metadata inspection
- [x] Dataset and subset selection
- [x] Image loading and preprocessing
- [x] Baseline CNN embedding extraction smoke test
- [x] MOA-balanced baseline training subset
- [x] Baseline supervised CNN training loop
- [x] Frozen pretrained representation benchmark
- [x] Linear probe on frozen embeddings
- [x] Perturbation-level fingerprint aggregation
- [x] Phenotypic nearest-neighbor search
- [x] Morphology-chemistry disagreement analysis
- [x] Final figures and interpretation

## References

- Cell Painting Gallery: an open resource for image-based profiling
- JUMP Cell Painting Consortium
- Cell Painting: a decade of discovery and innovation in cellular imaging
- Image-based profiling for drug discovery
