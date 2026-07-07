from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CHANNEL_COLUMNS = (
    "url_rna",
    "url_mito",
    "url_agp",
    "url_er",
    "url_dna",
)


def select_moa_balanced_subset(
    metadata: pd.DataFrame,
    mechanisms: list[str],
    sites_per_compound: int = 2,
    channel_columns: tuple[str, ...] = DEFAULT_CHANNEL_COLUMNS,
) -> pd.DataFrame:
    """Select a small MOA-balanced image subset for image-loading smoke tests."""
    required = {
        "image_record_id",
        "perturbation_id",
        "compound_name",
        "mechanism_of_action",
        "well",
        "site",
        *channel_columns,
    }
    missing = required.difference(metadata.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Metadata is missing required columns: {missing_text}")

    selected_frames = []
    labeled = metadata.dropna(subset=["mechanism_of_action", "perturbation_id"]).copy()
    for mechanism in mechanisms:
        moa_frame = labeled.loc[labeled["mechanism_of_action"] == mechanism]
        compound_ids = sorted(moa_frame["perturbation_id"].unique())
        if len(compound_ids) < 2:
            raise ValueError(f"Mechanism has fewer than two compounds: {mechanism}")

        for compound_id in compound_ids[:2]:
            compound_frame = (
                moa_frame.loc[moa_frame["perturbation_id"] == compound_id]
                .sort_values(["well", "site"])
                .head(sites_per_compound)
            )
            selected_frames.append(compound_frame)

    subset = pd.concat(selected_frames, ignore_index=True)
    return subset.sort_values(
        ["mechanism_of_action", "compound_name", "well", "site"]
    ).reset_index(drop=True)


def create_channel_manifest(
    subset: pd.DataFrame,
    channel_columns: tuple[str, ...] = DEFAULT_CHANNEL_COLUMNS,
    image_root: str = "data/raw/images/cpg0002-jump-scope/smoke_test",
) -> pd.DataFrame:
    """Convert a row-per-site subset into a row-per-channel download manifest."""
    rows = []
    for _, record in subset.iterrows():
        for channel_column in channel_columns:
            channel = channel_column.removeprefix("url_")
            rows.append(
                {
                    "image_record_id": record["image_record_id"],
                    "perturbation_id": record["perturbation_id"],
                    "compound_name": record["compound_name"],
                    "mechanism_of_action": record["mechanism_of_action"],
                    "well": record["well"],
                    "site": record["site"],
                    "channel": channel,
                    "source_url": record[channel_column],
                    "local_path": f"{image_root}/{record['image_record_id']}_{channel}.tif",
                }
            )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a small MOA-balanced cpg0002 image download manifest."
    )
    parser.add_argument("--metadata", required=True, help="Normalized metadata CSV.")
    parser.add_argument("--output", required=True, help="Output image manifest CSV.")
    parser.add_argument(
        "--mechanism",
        action="append",
        required=True,
        help="Mechanism-of-action label to include. Repeat for multiple mechanisms.",
    )
    parser.add_argument("--sites-per-compound", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = pd.read_csv(args.metadata)
    subset = select_moa_balanced_subset(
        metadata,
        mechanisms=args.mechanism,
        sites_per_compound=args.sites_per_compound,
    )
    manifest = create_channel_manifest(subset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(
        f"Wrote {len(manifest)} channel downloads from {subset['image_record_id'].nunique()} image records to {output}"
    )


if __name__ == "__main__":
    main()

