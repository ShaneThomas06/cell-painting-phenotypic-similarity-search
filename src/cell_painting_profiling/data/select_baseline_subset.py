from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cell_painting_profiling.data.select_image_subset import (
    DEFAULT_CHANNEL_COLUMNS,
    create_channel_manifest,
)


REQUIRED_COLUMNS = {
    "image_record_id",
    "perturbation_id",
    "compound_name",
    "mechanism_of_action",
    "well",
    "site",
    *DEFAULT_CHANNEL_COLUMNS,
}


def validate_metadata_columns(metadata: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(metadata.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Metadata is missing required columns: {missing_text}")


def summarize_mechanism_availability(
    metadata: pd.DataFrame,
    min_sites_per_compound: int = 4,
) -> pd.DataFrame:
    """Summarize which mechanisms can support a balanced baseline subset."""
    validate_metadata_columns(metadata)
    labeled = metadata.dropna(subset=["mechanism_of_action", "perturbation_id"]).copy()
    compound_counts = (
        labeled.groupby(["mechanism_of_action", "perturbation_id", "compound_name"])
        .agg(num_image_records=("image_record_id", "nunique"))
        .reset_index()
    )
    compound_counts["has_enough_sites"] = (
        compound_counts["num_image_records"] >= min_sites_per_compound
    )
    summary = (
        compound_counts.groupby("mechanism_of_action")
        .agg(
            eligible_compounds=("has_enough_sites", "sum"),
            total_compounds=("perturbation_id", "nunique"),
            total_image_records=("num_image_records", "sum"),
        )
        .reset_index()
        .sort_values(
            ["eligible_compounds", "total_image_records", "mechanism_of_action"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    return summary


def select_baseline_moa_subset(
    metadata: pd.DataFrame,
    max_mechanisms: int = 8,
    compounds_per_mechanism: int = 2,
    sites_per_compound: int = 4,
    mechanisms: list[str] | None = None,
) -> pd.DataFrame:
    """Select a larger MOA-balanced subset for baseline CNN training."""
    validate_metadata_columns(metadata)
    labeled = metadata.dropna(subset=["mechanism_of_action", "perturbation_id"]).copy()

    availability = summarize_mechanism_availability(
        labeled,
        min_sites_per_compound=sites_per_compound,
    )
    eligible_mechanisms = availability.loc[
        availability["eligible_compounds"] >= compounds_per_mechanism,
        "mechanism_of_action",
    ].tolist()

    if mechanisms is not None:
        requested = list(dict.fromkeys(mechanisms))
        missing = sorted(set(requested).difference(eligible_mechanisms))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Requested mechanisms are not eligible: {missing_text}")
        selected_mechanisms = requested
    else:
        selected_mechanisms = eligible_mechanisms[:max_mechanisms]

    if not selected_mechanisms:
        raise ValueError("No eligible mechanisms found for baseline subset selection")

    selected_frames = []
    for mechanism in selected_mechanisms:
        moa_frame = labeled.loc[labeled["mechanism_of_action"] == mechanism].copy()
        compound_counts = (
            moa_frame.groupby(["perturbation_id", "compound_name"])
            .agg(num_image_records=("image_record_id", "nunique"))
            .reset_index()
            .sort_values(["compound_name", "perturbation_id"])
        )
        eligible_compounds = compound_counts.loc[
            compound_counts["num_image_records"] >= sites_per_compound,
            "perturbation_id",
        ].tolist()
        for compound_id in eligible_compounds[:compounds_per_mechanism]:
            compound_frame = (
                moa_frame.loc[moa_frame["perturbation_id"] == compound_id]
                .sort_values(["well", "site", "image_record_id"])
                .head(sites_per_compound)
            )
            selected_frames.append(compound_frame)

    subset = pd.concat(selected_frames, ignore_index=True)
    return subset.sort_values(
        ["mechanism_of_action", "compound_name", "well", "site", "image_record_id"]
    ).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a larger MOA-balanced cpg0002 baseline image manifest."
    )
    parser.add_argument("--metadata", required=True, help="Normalized metadata CSV.")
    parser.add_argument("--output", required=True, help="Output image manifest CSV.")
    parser.add_argument("--max-mechanisms", type=int, default=8)
    parser.add_argument("--compounds-per-mechanism", type=int, default=2)
    parser.add_argument("--sites-per-compound", type=int, default=4)
    parser.add_argument(
        "--mechanism",
        action="append",
        help="Mechanism-of-action label to include. Repeat for multiple mechanisms.",
    )
    parser.add_argument(
        "--image-root",
        default="data/raw/images/cpg0002-jump-scope/baseline_training",
        help="Local image directory to write into the manifest.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = pd.read_csv(args.metadata)
    subset = select_baseline_moa_subset(
        metadata,
        max_mechanisms=args.max_mechanisms,
        compounds_per_mechanism=args.compounds_per_mechanism,
        sites_per_compound=args.sites_per_compound,
        mechanisms=args.mechanism,
    )
    manifest = create_channel_manifest(subset, image_root=args.image_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(
        f"Wrote {len(manifest)} channel downloads from "
        f"{subset['image_record_id'].nunique()} image records, "
        f"{subset['perturbation_id'].nunique()} compounds, and "
        f"{subset['mechanism_of_action'].nunique()} mechanisms to {output}"
    )


if __name__ == "__main__":
    main()
