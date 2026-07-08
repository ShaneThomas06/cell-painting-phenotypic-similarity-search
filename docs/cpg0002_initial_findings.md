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

## Multi-Channel CNN Smoke Test

A multi-channel dataset class was added to load one image site as a stacked fluorescence tensor.

Tensor contract:

```text
one image site -> 5 fluorescence channels -> [5, 224, 224] tensor
```

Channels used:

```text
RNA
Mito
AGP
ER
DNA
```

A ResNet18 forward-pass smoke test was run with `pretrained=False` to avoid downloading model weights. The goal was not training; it was to verify that real downloaded TIFFs can be loaded, resized, stacked, and passed through a CNN.

Observed forward-pass result:

```text
image records: 16
channel rows: 80
input shape: [4, 5, 224, 224]
mechanism classes: 4
output shape: [4, 4]
```

This confirms the project is ready for the next stage: extracting CNN embeddings from the smoke-test subset, then scaling the workflow to a larger MOA-balanced sample.

## First Embedding Retrieval Smoke Test

The smoke-test images were converted into CNN embeddings, then averaged into one compound-level fingerprint per compound.

Plain-language interpretation:

```text
image pixels -> CNN numbers -> compound fingerprint -> nearest compound search
```

Observed output:

```text
16 image-level embeddings
8 compound-level fingerprints
24 nearest-neighbor rows
top-1 shared mechanism rate: 0.0000
top-3 row-level shared mechanism rate: 0.1667
```

This result should not be interpreted as a real biological signal yet. The CNN used random, untrained weights for this smoke test, so weak mechanism recovery is expected. The important result is technical: the project can now load real Cell Painting images, convert them into embeddings, aggregate those embeddings by compound, and run a first phenotypic similarity search.

## Baseline Training Subset

A larger MOA-balanced manifest was generated for the first supervised CNN baseline.

Selection rule:

```text
8 mechanism-of-action classes
2 compounds per mechanism
4 image sites per compound
5 fluorescence channels per image site
```

Observed manifest:

```text
64 image records
320 channel-level image downloads
16 compounds
8 mechanisms
```

This subset is intentionally balanced so the baseline CNN sees the same number of compounds and image sites for each mechanism. The next step is to download and validate these images, then train a small MOA classifier and use its penultimate layer as the improved embedding.

## Baseline Image Download And Validation

The baseline training manifest was downloaded locally and validated.

Validation result:

```text
320 / 320 channel image files present
0 validation failures
all images opened successfully
all images were 1994 x 1994 16-bit TIFFs
```

This means the baseline training subset is now ready for the first supervised CNN training run. The next modeling step is to train a small ResNet18 classifier to predict mechanism-of-action from the 5-channel image tensors.

## First Trained Baseline Retrieval Result

A 3-epoch ResNet18 baseline was trained on the baseline subset and then reused as an embedding extractor.

Observed result:

```text
final training loss: 0.5388
final validation accuracy: 0.1250
64 trained image embeddings
16 trained compound fingerprints
top-1 shared mechanism rate: 0.0625
top-3 row-level shared mechanism rate: 0.0208
```

This result is close to chance and should be treated as a weak baseline. The useful conclusion is not that the model has discovered biology yet; it is that the project now supports the complete trained-model workflow: train CNN, extract trained embeddings, aggregate compound fingerprints, and evaluate phenotypic retrieval.

## Larger 12-Site Baseline Result

The image subset was expanded from 4 to 12 image sites per compound.

Observed result:

```text
192 image records
960 channel-level image files
0 image validation failures
final training loss: 0.4811
final validation accuracy: 0.1250
top-1 shared mechanism rate: 0.0625
queries with a shared-mechanism neighbor in top 3: 0.1875
```

Increasing image sites improved top-3 retrieval coverage slightly, but classification validation remained at chance. This suggests that replicate coverage alone is not enough; the next modeling improvement should focus on better representations, such as pretrained image features or stronger regularization.
