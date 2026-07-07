from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def get_embedding_columns(frame: pd.DataFrame) -> list[str]:
    """Return embedding feature columns."""
    return [column for column in frame.columns if column.startswith("embedding_")]


def aggregate_compound_fingerprints(embeddings: pd.DataFrame) -> pd.DataFrame:
    """Average image-level embeddings into compound-level fingerprints."""
    required = {"perturbation_id", "compound_name", "mechanism_of_action"}
    missing = required.difference(embeddings.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Embeddings table is missing required columns: {missing_text}")

    embedding_columns = get_embedding_columns(embeddings)
    if not embedding_columns:
        raise ValueError("Embeddings table has no embedding_* columns")

    group_columns = ["perturbation_id", "compound_name", "mechanism_of_action"]
    fingerprints = (
        embeddings.groupby(group_columns, dropna=False)[embedding_columns]
        .mean()
        .reset_index()
    )
    fingerprints["num_image_records"] = (
        embeddings.groupby(group_columns, dropna=False)["image_record_id"]
        .nunique()
        .to_numpy()
    )
    return fingerprints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate image embeddings into compound fingerprints.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    embeddings = pd.read_csv(args.embeddings)
    fingerprints = aggregate_compound_fingerprints(embeddings)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fingerprints.to_csv(output, index=False)
    print(f"Wrote {len(fingerprints)} compound fingerprints to {output}")


if __name__ == "__main__":
    main()
