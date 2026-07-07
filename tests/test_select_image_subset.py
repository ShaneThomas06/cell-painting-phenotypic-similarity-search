import pandas as pd

from cell_painting_profiling.data.select_image_subset import (
    create_channel_manifest,
    select_moa_balanced_subset,
)


def test_select_moa_balanced_subset_picks_two_compounds_per_moa():
    metadata = pd.DataFrame(
        {
            "image_record_id": [f"id_{i}" for i in range(6)],
            "perturbation_id": ["a", "a", "b", "b", "c", "c"],
            "compound_name": ["A", "A", "B", "B", "C", "C"],
            "mechanism_of_action": ["m1", "m1", "m1", "m1", "m2", "m2"],
            "well": ["A01", "A01", "A02", "A02", "A03", "A03"],
            "site": [1, 2, 1, 2, 1, 2],
            "url_rna": ["rna"] * 6,
            "url_mito": ["mito"] * 6,
            "url_agp": ["agp"] * 6,
            "url_er": ["er"] * 6,
            "url_dna": ["dna"] * 6,
        }
    )

    subset = select_moa_balanced_subset(metadata, mechanisms=["m1"], sites_per_compound=1)

    assert subset["perturbation_id"].nunique() == 2
    assert len(subset) == 2


def test_create_channel_manifest_expands_records_to_channels():
    subset = pd.DataFrame(
        {
            "image_record_id": ["id_1"],
            "perturbation_id": ["a"],
            "compound_name": ["A"],
            "mechanism_of_action": ["m1"],
            "well": ["A01"],
            "site": [1],
            "url_rna": ["s3://rna"],
            "url_mito": ["s3://mito"],
            "url_agp": ["s3://agp"],
            "url_er": ["s3://er"],
            "url_dna": ["s3://dna"],
        }
    )

    manifest = create_channel_manifest(subset)

    assert len(manifest) == 5
    assert set(manifest["channel"]) == {"rna", "mito", "agp", "er", "dna"}
