from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LOAD_DATA_RENAME_MAP = {
    "URL_OrigBrightField": "url_brightfield",
    "URL_OrigRNA": "url_rna",
    "URL_OrigMito": "url_mito",
    "URL_OrigAGP": "url_agp",
    "URL_OrigER": "url_er",
    "URL_OrigDNA": "url_dna",
    "Metadata_Well": "well",
    "Metadata_Site": "site",
    "Metadata_Plate": "plate",
}

COMPOUND_RENAME_MAP = {
    "broad_sample": "perturbation_id",
    "pert_iname": "compound_name",
    "moa": "mechanism_of_action",
    "pert_type": "perturbation_type",
    "smiles": "smiles",
    "InChIKey": "inchikey",
    "pubchem_cid": "pubchem_cid",
    "control_type": "control_type",
}


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required columns: {missing_text}")


def normalize_cpg0002_metadata(
    load_data: pd.DataFrame,
    platemap: pd.DataFrame,
    compound_metadata: pd.DataFrame,
    dataset_id: str = "cpg0002-jump-scope",
) -> pd.DataFrame:
    """Join cpg0002 load-data, platemap, and compound metadata tables."""
    require_columns(load_data, set(LOAD_DATA_RENAME_MAP), "load_data")
    require_columns(platemap, {"well_position", "broad_sample"}, "platemap")
    require_columns(compound_metadata, {"broad_sample", "pert_iname", "moa"}, "compound_metadata")

    load_data = load_data.rename(columns=LOAD_DATA_RENAME_MAP).copy()
    platemap = platemap.rename(columns={"well_position": "well"}).copy()
    compounds = compound_metadata.rename(columns=COMPOUND_RENAME_MAP).copy()

    merged = load_data.merge(platemap, on="well", how="left", validate="many_to_one")
    merged = merged.merge(
        compounds,
        left_on="broad_sample",
        right_on="perturbation_id",
        how="left",
        validate="many_to_one",
    )

    merged["dataset_id"] = dataset_id
    merged["image_record_id"] = (
        merged["dataset_id"].astype(str)
        + "__"
        + merged["plate"].astype(str)
        + "__"
        + merged["well"].astype(str)
        + "__site"
        + merged["site"].astype(str)
    )

    ordered_columns = [
        "image_record_id",
        "dataset_id",
        "plate",
        "well",
        "site",
        "perturbation_id",
        "compound_name",
        "mechanism_of_action",
        "perturbation_type",
        "control_type",
        "smiles",
        "inchikey",
        "pubchem_cid",
        "solvent",
        "url_brightfield",
        "url_rna",
        "url_mito",
        "url_agp",
        "url_er",
        "url_dna",
    ]
    available_columns = [column for column in ordered_columns if column in merged.columns]
    return merged[available_columns].sort_values(["plate", "well", "site"]).reset_index(drop=True)


def load_and_normalize_cpg0002_metadata(
    load_data_path: str | Path,
    platemap_path: str | Path,
    compound_metadata_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load local cpg0002 metadata files and return a normalized table."""
    load_data = pd.read_csv(load_data_path)
    platemap = pd.read_csv(platemap_path, sep="\t")
    compound_metadata = pd.read_csv(compound_metadata_path, sep="\t")
    normalized = normalize_cpg0002_metadata(load_data, platemap, compound_metadata)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(output_path, index=False)

    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize cpg0002 JUMP-Scope metadata into an image-record table."
    )
    parser.add_argument("--load-data", required=True, help="Path to load_data.csv.")
    parser.add_argument("--platemap", required=True, help="Path to JUMP-MOA platemap TSV.")
    parser.add_argument("--compound-metadata", required=True, help="Path to JUMP-MOA compound metadata TSV.")
    parser.add_argument("--output", required=True, help="Path for normalized metadata CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    normalized = load_and_normalize_cpg0002_metadata(
        load_data_path=args.load_data,
        platemap_path=args.platemap,
        compound_metadata_path=args.compound_metadata,
        output_path=args.output,
    )
    print(f"Wrote {len(normalized)} normalized image records to {args.output}")


if __name__ == "__main__":
    main()
