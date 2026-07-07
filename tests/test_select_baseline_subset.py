import pandas as pd

from cell_painting_profiling.data.select_baseline_subset import (
    select_baseline_moa_subset,
    summarize_mechanism_availability,
)
from cell_painting_profiling.data.select_image_subset import create_channel_manifest


def make_metadata() -> pd.DataFrame:
    rows = []
    for mechanism in ["m1", "m2", "m3"]:
        for compound_index in range(3):
            compound_id = f"{mechanism}_compound_{compound_index}"
            for site in range(5):
                rows.append(
                    {
                        "image_record_id": f"{compound_id}_site_{site}",
                        "perturbation_id": compound_id,
                        "compound_name": compound_id.upper(),
                        "mechanism_of_action": mechanism,
                        "well": f"A0{compound_index + 1}",
                        "site": site + 1,
                        "url_rna": "s3://rna",
                        "url_mito": "s3://mito",
                        "url_agp": "s3://agp",
                        "url_er": "s3://er",
                        "url_dna": "s3://dna",
                    }
                )
    return pd.DataFrame(rows)


def test_summarize_mechanism_availability_counts_eligible_compounds():
    summary = summarize_mechanism_availability(
        make_metadata(),
        min_sites_per_compound=4,
    )

    assert set(summary["mechanism_of_action"]) == {"m1", "m2", "m3"}
    assert set(summary["eligible_compounds"]) == {3}


def test_select_baseline_moa_subset_is_balanced_by_mechanism_and_compound():
    subset = select_baseline_moa_subset(
        make_metadata(),
        max_mechanisms=2,
        compounds_per_mechanism=2,
        sites_per_compound=4,
    )

    assert subset["mechanism_of_action"].nunique() == 2
    assert subset["perturbation_id"].nunique() == 4
    assert len(subset) == 16
    assert set(subset.groupby("mechanism_of_action")["perturbation_id"].nunique()) == {2}
    assert set(subset.groupby("perturbation_id")["image_record_id"].nunique()) == {4}


def test_baseline_manifest_uses_training_image_root():
    subset = select_baseline_moa_subset(
        make_metadata(),
        max_mechanisms=1,
        compounds_per_mechanism=2,
        sites_per_compound=1,
    )
    manifest = create_channel_manifest(
        subset,
        image_root="data/raw/images/cpg0002-jump-scope/baseline_training",
    )

    assert len(manifest) == 10
    assert manifest["local_path"].str.contains("baseline_training").all()
