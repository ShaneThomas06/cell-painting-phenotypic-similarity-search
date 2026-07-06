# Dataset Reconnaissance

## Purpose

This project needs a Cell Painting dataset that supports a phenotypic similarity search workflow rather than a standard classification benchmark. The dataset must provide enough image data for CNN feature extraction and enough perturbation annotations to evaluate whether morphologically similar perturbations share mechanism, target, pathway, or compound-class information.

## Selection Criteria

A strong first dataset should satisfy four criteria:

1. **Image availability**: raw or processed Cell Painting images can be accessed without special permissions.
2. **Annotation richness**: perturbations have mechanism, target, pathway, compound, gene, or chemical identifiers.
3. **Feasible first scope**: a subset can be downloaded and processed locally without requiring full-dataset infrastructure.
4. **Drug-discovery relevance**: the analysis can produce interpretable perturbation-neighbor examples, not just benchmark accuracy.

## Public Data Landscape

The Cell Painting Gallery is the main public access point for large Cell Painting datasets. It is hosted on AWS Open Data and includes microscopy images, extracted features, and metadata. The gallery documentation states that datasets can be browsed and downloaded without an AWS account using AWS CLI or Quilt, and that users should inspect sizes before downloading.

The JUMP Cell Painting resources are especially relevant because they were created to support large-scale morphological profiling of chemical and genetic perturbations. The JUMP Hub also provides notebooks and interactive tools for retrieving profiles, adding metadata, calculating phenotypic activity, displaying perturbation images, and exploring perturbation clusters.

## Candidate Datasets

### 1. JUMP Cell Painting Full Dataset (`cpg0016-jump`)

**Why it is attractive**

- Largest and most important Cell Painting resource for this project direction.
- Includes chemical and genetic perturbations.
- Strong fit for phenotypic similarity search and mechanism-discovery framing.
- High credibility because it is a community-scale pharma/academic consortium dataset.

**Risks**

- Very large image volume, not suitable for immediate full download.
- Requires careful subset selection and metadata handling.
- Better as the long-term target than the first local experiment.

**Role in this project**

Use as the primary conceptual dataset, but begin with a small, metadata-rich subset.

### 2. JUMP Pilot (`cpg0000-jump-pilot`)

**Why it is attractive**

- Smaller than the full JUMP dataset.
- Includes hundreds of compounds and genes.
- Useful for developing the image loading, embedding, and retrieval pipeline.
- Better first implementation target than full JUMP.

**Risks**

- Smaller perturbation space may limit mechanism-of-action diversity.
- Still large enough that image downloading must be selective.

**Role in this project**

Strong first development target if metadata supports mechanism or target labels.

### 3. JUMP-MOA Plate / Scope Dataset (`cpg0002-jump-scope`)

**Why it is attractive**

- Focused set of 90 compounds from a JUMP mechanism-of-action plate.
- Very aligned with the mechanism-recovery objective.
- Useful for a proof-of-concept retrieval benchmark.

**Risks**

- Limited compound count.
- Dataset was generated across microscope settings, so technical variation may be a major confounder.

**Role in this project**

Best candidate for an early mechanism-recovery demo if image and metadata paths are straightforward.

### 4. Bioactive Compound Profiling (`cpg0012-wawer-bioactivecompoundprofiling`)

**Why it is attractive**

- Large bioactive-compound dataset.
- Strong drug-discovery relevance.
- Good candidate for morphology-chemistry disagreement analysis.

**Risks**

- Metadata harmonization may be more work.
- Full image volume is too large for local-first development.

**Role in this project**

Good second-stage dataset once the retrieval pipeline works.

### 5. MOTIVE (`cpg0034-arevalo-su-motive`)

**Why it is attractive**

- Directly links Cell Painting features to drug-target interaction relationships.
- Excellent for mechanism and target interpretation.
- Small enough to be practical.

**Risks**

- Does not contain images, so it cannot be the main CNN image-handling dataset.

**Role in this project**

Use as an annotation/interpretation companion or optional downstream graph-analysis extension, not as the core image dataset.

## Recommended Direction

The project should proceed in two stages:

### Stage 1: Proof-of-Concept Retrieval

Use a focused JUMP-related subset, ideally the JUMP-MOA plate or JUMP pilot, to build the end-to-end workflow:

```text
images -> CNN embeddings -> perturbation fingerprints -> nearest-neighbor search -> mechanism/target recovery
```

This stage should prioritize a subset with clear labels and manageable download size.

### Stage 2: Discovery-Oriented Expansion

After the pipeline works, expand to a larger chemical perturbation dataset such as the full JUMP subset or the Wawer bioactive-compound profiling dataset. This is where the project can emphasize morphology-chemistry disagreement and unexpected phenotypic neighbors.

## Decision For The Next Implementation Step

Before downloading images, inspect metadata for these candidates in order:

1. `cpg0002-jump-scope`
2. `cpg0000-jump-pilot`
3. `cpg0012-wawer-bioactivecompoundprofiling`
4. `cpg0016-jump`

The first dataset selected for image download should be the smallest candidate that supports mechanism or target recovery analysis.

Initial metadata reconnaissance has begun for cpg0002-jump-scope; see [cpg0002 initial findings](cpg0002_initial_findings.md).

## Non-Negotiables

- Do not download full multi-terabyte datasets locally.
- Start with metadata and a small image subset.
- Keep raw images out of Git history.
- Treat batch, microscope, plate, and site labels as possible confounders.
- Report morphology-based similarity as hypothesis-generating, not proof of direct mechanism.

## Sources To Cite

- Cell Painting Gallery documentation and Registry of Open Data entry.
- JUMP Cell Painting Hub.
- JUMP Cell Painting datasets repository.
- Dataset-specific papers for whichever dataset is selected for the first experiment.

