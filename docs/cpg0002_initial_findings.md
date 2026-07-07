# cpg0002 JUMP-Scope Initial Findings

## Why This Dataset Is A Strong First Target

`cpg0002-jump-scope` is a focused Cell Painting Gallery dataset built around the JUMP-MOA compound plate. It is a good first target because it is small enough to reason about, contains compound mechanism labels, and provides image path metadata that can be joined to plate annotations.

## Metadata Files Inspected

Small metadata files downloaded locally for inspection:

```text
data/raw/cpg0002-jump-scope/metadata/JUMP-MOA_compound_metadata.tsv
data/raw/cpg0002-jump-scope/metadata/JUMP-MOA_compound_platemap.txt
data/raw/cpg0002-jump-scope/load_data/BRO0117059_20X/load_data.csv
```

These files are ignored by Git because they live under `data/raw/`.

## Compound Metadata

Source:

```text
s3://cellpainting-gallery/cpg0002-jump-scope/source_4/workspace/metadata/external_metadata/JUMP-MOA_compound_metadata.tsv
```

Observed shape:

```text
91 rows x 8 columns
```

Columns:

```text
broad_sample
InChIKey
pert_iname
pubchem_cid
moa
pert_type
control_type
smiles
```

Important finding:

```text
47 unique mechanism-of-action labels
```

This supports the core retrieval evaluation: whether nearest phenotypic neighbors share `moa`.

## Platemap Metadata

Source:

```text
s3://cellpainting-gallery/cpg0002-jump-scope/source_4/workspace/metadata/platemaps/Scope1_Yokogawa_US_20X_6Ch_BRO0117059/platemap/JUMP-MOA_compound_platemap.txt
```

Observed shape:

```text
384 rows x 3 columns
```

Columns:

```text
well_position
broad_sample
solvent
```

Important finding:

```text
91 broad_sample values overlap with the compound metadata table
```

This confirms that the platemap can link wells to compound/MOA annotations.

## Load Data

Source:

```text
s3://cellpainting-gallery/cpg0002-jump-scope/source_4/workspace/load_data_csv/2020_11_16_Scope1_YokogawaUS/BRO0117059_20X/load_data.csv
```

Observed shape:

```text
3454 rows x 9 columns
```

Columns:

```text
URL_OrigBrightField
URL_OrigRNA
URL_OrigMito
URL_OrigAGP
URL_OrigER
URL_OrigDNA
Metadata_Well
Metadata_Site
Metadata_Plate
```

Important finding:

The load-data file contains S3 image paths for six channels plus well, site, and plate metadata. It does not contain compound IDs directly, so the join path is:

```text
load_data.Metadata_Well -> platemap.well_position -> compound_metadata.broad_sample
```

## Initial Technical Interpretation

This dataset is viable for the first proof-of-concept:

```text
six-channel image URLs -> well/site image records -> compound IDs -> MOA labels
```

The main confounder is microscope/scope variation. The first CNN experiment should start with one plate/scope setting, such as `BRO0117059_20X`, before expanding across microscopes.

## Next Implementation Step

Build a normalization script for `cpg0002-jump-scope` that creates an analysis-ready metadata table:

```text
image record
channel URLs
well
site
plate
broad_sample
compound name
MOA
SMILES
```

The output should live under `data/processed/` and remain ignored by Git.

## Normalization Script Result

The normalization script successfully joined the selected load-data table, platemap, and compound metadata.

Command pattern:

```bash
python -m cell_painting_profiling.data.normalize_cpg0002 \
  --load-data data/raw/cpg0002-jump-scope/load_data/BRO0117059_20X/load_data.csv \
  --platemap data/raw/cpg0002-jump-scope/metadata/JUMP-MOA_compound_platemap.txt \
  --compound-metadata data/raw/cpg0002-jump-scope/metadata/JUMP-MOA_compound_metadata.tsv \
  --output data/processed/cpg0002-jump-scope/BRO0117059_20X_normalized_metadata.csv
```

Observed output:

```text
3454 image records
20 normalized columns
90 compounds with image records
47 mechanism-of-action labels
216 image records without compound/MOA annotations
```

The missing annotations should be treated explicitly. The first retrieval benchmark should filter to rows where `mechanism_of_action` is present, while retaining unannotated/control rows only for quality-control checks.

## Smoke-Test Image Subset

A tiny MOA-balanced image subset was selected from the normalized metadata for image-loading validation.

Selection rule:

```text
4 mechanism-of-action classes
2 compounds per mechanism
2 image sites per compound
5 fluorescence channels per site
```

Mechanisms selected:

```text
HDAC inhibitor
EGFR inhibitor
CDK inhibitor
JAK inhibitor
```

Compounds selected:

```text
HDAC inhibitor: RGFP966, romidepsin
EGFR inhibitor: CP-724714, neratinib
CDK inhibitor: AMG-925, THZ1
JAK inhibitor: filgotinib, ruxolitinib
```

Observed manifest:

```text
16 image records
80 channel-level image downloads
```

Validation result:

```text
80 / 80 files opened successfully
all images were 1994 x 1994 16-bit TIFFs
```

This confirms that the cpg0002 metadata, image URLs, local paths, and channel-level download logic are ready for a first multi-channel image loader.
