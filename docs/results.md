# Representation Benchmark Results

## Objective

This analysis compares image representation strategies for mechanism-aware phenotypic retrieval from Cell Painting images. The goal is to evaluate whether learned image embeddings can recover compounds with shared mechanism-of-action labels under a compound-holdout setting.

## Dataset And Split

The benchmark uses the `cpg0002-jump-scope` 12-site baseline subset:

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

Validation uses a compound-holdout split. For each mechanism, one compound is used for training and the other compound is held out for validation. This evaluates whether representations generalize across compounds within the same mechanism rather than memorizing image sites from the same compound.

## Compared Representations

Three representation strategies were compared:

```text
random_12site_finetuned
pretrained_augmented_finetuned
frozen_pretrained
```

The random model uses ResNet18 initialized from scratch. The pretrained fine-tuned model uses ImageNet-pretrained ResNet18 weights, channel normalization, and light flip or rotation augmentation. The frozen pretrained model uses ImageNet-pretrained ResNet18 as a fixed feature extractor with dataset-specific channel statistics.

## Summary Table

| Experiment | Validation accuracy | Top-1 shared MOA | Top-3 row shared MOA | Query has shared MOA in top 3 | Linear probe image accuracy | Linear probe compound accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Random ResNet18 fine-tuned | 0.1250 | 0.0625 | 0.0625 | 0.1875 | NA | NA |
| Pretrained ResNet18 fine-tuned | 0.1667 | 0.1875 | 0.0833 | 0.2500 | NA | NA |
| Frozen pretrained ResNet18 | NA | 0.0625 | 0.1042 | 0.3125 | 0.1667 | 0.1250 |

The full machine-readable summary is available at:

```text
reports/tables/model_comparison_summary.csv
```

The retrieval comparison figure is available at:

```text
reports/figures/model_comparison_retrieval.png
```

## Interpretation

The random fine-tuned model reached chance-level validation accuracy and weak retrieval. Increasing image sites improved coverage slightly, but did not solve compound-holdout generalization.

The pretrained fine-tuned model gave the strongest top-1 retrieval result. This suggests that pretrained visual features can sharpen nearest-neighbor matches when fine-tuned on the Cell Painting subset.

The frozen pretrained model gave the strongest top-3 query coverage. This suggests that fixed pretrained features may preserve broader morphology structure, even when they do not provide the best nearest neighbor for every query.

The linear probe on frozen embeddings achieved weak image-level classification performance and chance-level compound-level performance. This indicates that the frozen embeddings contain limited mechanism signal, but the signal is not strong enough to classify held-out compounds reliably.

## Visual Diagnostics

Compound-level UMAP projections were generated for the three representation strategies:

```text
reports/figures/compound_embedding_umap.png
```

The UMAP plots are diagnostic rather than confirmatory. They provide a visual summary of whether compounds with the same mechanism occupy nearby regions of representation space. The current projections do not show strong mechanism-level separation, which is consistent with the retrieval metrics.

Representative retrieval panels were generated from the pretrained fine-tuned model:

```text
reports/figures/retrieval_example_panel.png
reports/tables/retrieval_example_cases.csv
```

The retrieval examples include one case where the rank-1 neighbor shares the query mechanism and one case where the rank-1 neighbor differs but a shared-mechanism compound appears at rank 3. These examples illustrate the current behavior of the retrieval model: useful local matches occur, but mechanism recovery is not consistent.

## Morphology-Chemistry Comparison

Chemical similarity was computed from SMILES using RDKit Morgan fingerprints and Tanimoto similarity. Morphology similarity was computed from pretrained fine-tuned ResNet18 compound fingerprints using cosine similarity. This analysis included 15 compounds with valid SMILES and produced 105 compound pairs.

The machine-readable outputs are available at:

```text
reports/tables/morphology_chemistry_pairwise.csv
reports/tables/morphology_chemistry_cases.csv
```

The summary figure is available at:

```text
reports/figures/morphology_chemistry_similarity.png
```

The strongest morphology-chemistry disagreement case was BMS-863233 and KH-CB19. Both compounds are annotated as CDC inhibitors and showed high morphology similarity, but their Morgan fingerprint similarity was low. This is a useful positive example for phenotypic profiling: compounds can converge on a similar cellular phenotype without being chemically close.

The strongest same-mechanism low-morphology case was ponatinib and GNF-5, both annotated as Bcr-Abl kinase inhibitors. This illustrates the opposite limitation: shared mechanism labels do not guarantee similar Cell Painting profiles in a small image subset.

## Main Conclusion

The benchmark supports a representation-centered interpretation. The pipeline is technically complete, but the current dataset scale and generic pretrained features limit biological mechanism recovery. Small supervised fine-tuning is not sufficient under compound-holdout validation. The morphology-chemistry comparison adds a second interpretation layer by showing where cellular phenotype similarity agrees or disagrees with molecular structure similarity.

The next meaningful improvements require one of the following:

```text
larger compound diversity
bioimage-specific pretrained models
self-supervised Cell Painting representation learning
classical morphology-feature benchmarking with CellProfiler-style features
more compute for larger representation learning experiments
```

These are substantive project extensions. The current baseline phase establishes a reproducible comparison framework for evaluating them.
