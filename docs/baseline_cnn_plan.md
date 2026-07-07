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
