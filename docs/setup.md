# Setup

Use Python 3.10, 3.11, 3.12, or 3.13 for this project. The deep-learning stack depends on PyTorch and torchvision, which may lag behind the newest Python releases.

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Verify

```bash
pytest
```

## Data Policy

Raw Cell Painting image data is large and should stay outside Git history. Keep downloaded data under `data/raw/`, which is ignored except for placeholder files.

The tracked `reports/` files contain final figures and compact result tables. The larger raw images, processed image manifests, and embedding tables are generated locally during analysis.

## Metadata Inspection

After downloading or preparing a Cell Painting metadata table:

```bash
inspect-cell-painting-metadata --metadata data/raw/metadata.csv --output reports/tables/metadata_summary.csv
```

## Reproduce Final Result Figures

The following commands regenerate the final comparison artifacts after the local processed data files have been created.

Generate the representation comparison summary:

```bash
compare-cell-painting-experiments --tables-dir reports/tables --output-table reports/tables/model_comparison_summary.csv --output-figure reports/figures/model_comparison_retrieval.png
```

Generate UMAP and retrieval example figures:

```bash
generate-cell-painting-result-figures --manifest data/processed/cpg0002-jump-scope/BRO0117059_20X_baseline_12site_image_manifest.csv --neighbors reports/tables/baseline_12site_pretrained_augmented_resnet18_nearest_neighbors.csv --umap-output reports/figures/compound_embedding_umap.png --retrieval-output reports/figures/retrieval_example_panel.png --retrieval-table reports/tables/retrieval_example_cases.csv
```

Generate morphology-chemistry comparison artifacts:

```bash
compare-morphology-chemistry --fingerprints data/processed/cpg0002-jump-scope/BRO0117059_20X_baseline_12site_pretrained_augmented_compound_fingerprints.csv --metadata data/processed/cpg0002-jump-scope/BRO0117059_20X_normalized_metadata.csv --output-pairwise reports/tables/morphology_chemistry_pairwise.csv --output-cases reports/tables/morphology_chemistry_cases.csv --output-figure reports/figures/morphology_chemistry_similarity.png --manifest data/processed/cpg0002-jump-scope/BRO0117059_20X_baseline_12site_image_manifest.csv --output-case-panel reports/figures/morphology_chemistry_case_studies.png
```
