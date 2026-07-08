# Baseline CNN Training Plan

## Purpose

The tiny smoke-test subset proved that the image pipeline works. The next baseline needs a larger, balanced subset so a CNN can begin learning real morphology patterns instead of only proving that files can be opened.

Simple version:

```text
balanced images -> train CNN to predict MOA -> extract better embeddings -> rerun similarity search
```

## Baseline Subset

The first training subset is selected from `cpg0002-jump-scope` using one plate/scope setting: `BRO0117059_20X`.

Default selection:

```text
8 mechanism-of-action classes
2 compounds per mechanism
4 image sites per compound
5 fluorescence channels per image site
```

Expected manifest size:

```text
64 image records
320 channel-level image downloads
16 compounds
8 mechanisms
```

This is bigger than the smoke test, but still small enough for local development.

## Why Balance Matters

If one mechanism has many more images than another, a model can look strong for the wrong reason: it may mostly learn the most common class. A balanced subset makes the first baseline fairer and easier to explain.

This subset balances three things:

- same number of mechanisms
- same number of compounds per mechanism
- same number of image sites per compound

## Initial Local Manifest

The first generated baseline manifest used:

```text
max mechanisms: 8
compounds per mechanism: 2
sites per compound: 4
```

Selected mechanisms:

```text
Aurora kinase inhibitor
BCL inhibitor
Bcr-Abl kinase inhibitor
CDC inhibitor
CDK inhibitor
CHK inhibitor
DNA inhibitor
EGFR inhibitor
```

The manifest is written locally under `data/processed/` and ignored by Git.

## Next Step

Download the 320 channel images, validate that they open correctly, then train a small ResNet18 MOA classifier. After training, the layer before the classifier head will be used as the improved image embedding.

## Baseline ResNet18 Training Smoke Test

A supervised ResNet18 training loop was added for mechanism-of-action prediction from 5-channel Cell Painting tensors.

Smoke-test command pattern:

```bash
python -m cell_painting_profiling.training.train \
  --manifest data/processed/cpg0002-jump-scope/BRO0117059_20X_baseline_training_image_manifest.csv \
  --metrics-output reports/tables/baseline_resnet18_smoke_metrics.json \
  --model-output models/baseline_resnet18_smoke.pt \
  --image-size 96 \
  --batch-size 4 \
  --epochs 1 \
  --max-train-batches 1 \
  --max-val-batches 1
```

Observed smoke-test result:

```text
device: CPU
mechanism classes: 8
train image records: 32
validation image records: 32
one training batch completed
one validation batch completed
model checkpoint saved locally
metrics JSON saved locally
```

This confirms that the baseline training path can load real TIFFs, assign mechanism labels, train a 5-channel ResNet18 classifier, evaluate it, and save outputs. The reported smoke-test accuracy should not be interpreted biologically because the run used only one train batch and one validation batch.

## First 3-Epoch Baseline Result

A full baseline run was completed on the 64-image-record subset using compound-holdout validation.

Training setup:

```text
encoder: ResNet18
input channels: 5
image size: 96 x 96
epochs: 3
batch size: 8
training compounds: 8
validation compounds: 8
training image records: 32
validation image records: 32
```

Final epoch result:

```text
training loss: 0.5388
validation accuracy: 0.1250
validation balanced accuracy: 0.1250
validation macro F1: 0.0278
```

The training loss dropped, but validation accuracy stayed at chance level for 8 classes. In simple terms, the model learned patterns from the specific training compounds but did not yet generalize to the held-out compounds.

Trained-checkpoint retrieval result:

```text
64 image embeddings
16 compound fingerprints
48 nearest-neighbor rows
top-1 shared mechanism rate: 0.0625
top-3 row-level shared mechanism rate: 0.0208
queries with a shared-mechanism neighbor in top 3: 0.0625
```

This is an honest weak baseline. It shows the end-to-end trained-model workflow works, but the current subset is too small for strong biological retrieval. The next improvement should increase replicate coverage and/or use pretrained biological image features before expecting meaningful mechanism recovery.
