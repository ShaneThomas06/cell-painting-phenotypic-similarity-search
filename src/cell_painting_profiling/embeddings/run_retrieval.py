from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cell_painting_profiling.embeddings.aggregate import get_embedding_columns
from cell_painting_profiling.embeddings.retrieval import build_neighbor_table


def run_compound_retrieval(fingerprints: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Build a compound-level nearest-neighbor table."""
    embedding_columns = get_embedding_columns(fingerprints)
    if not embedding_columns:
        raise ValueError("Fingerprints table has no embedding_* columns")

    metadata = fingerprints[
        ["perturbation_id", "compound_name", "mechanism_of_action"]
    ].copy()
    embeddings = fingerprints[embedding_columns].to_numpy()
    neighbors = build_neighbor_table(
        embeddings,
        embeddings,
        metadata,
        metadata,
        query_id_column="perturbation_id",
        reference_id_column="perturbation_id",
        k=k + 1,
    )
    neighbors = neighbors.loc[neighbors["query_id"] != neighbors["neighbor_id"]].copy()
    neighbors["rank"] = neighbors.groupby("query_id").cumcount() + 1
    neighbors = neighbors.loc[neighbors["rank"] <= k].reset_index(drop=True)

    query_metadata = metadata.add_prefix("query_")
    neighbor_metadata = metadata.add_prefix("neighbor_")
    neighbors = neighbors.merge(
        query_metadata,
        left_on="query_id",
        right_on="query_perturbation_id",
        how="left",
    )
    neighbors = neighbors.merge(
        neighbor_metadata,
        left_on="neighbor_id",
        right_on="neighbor_perturbation_id",
        how="left",
    )
    neighbors["shared_mechanism"] = (
        neighbors["query_mechanism_of_action"]
        == neighbors["neighbor_mechanism_of_action"]
    )
    return neighbors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run compound-level phenotypic retrieval.")
    parser.add_argument("--fingerprints", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fingerprints = pd.read_csv(args.fingerprints)
    neighbors = run_compound_retrieval(fingerprints, k=args.k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    neighbors.to_csv(output, index=False)
    print(f"Wrote {len(neighbors)} nearest-neighbor rows to {output}")


if __name__ == "__main__":
    main()

