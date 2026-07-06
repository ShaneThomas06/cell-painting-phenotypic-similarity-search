from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cell_painting_profiling.data.dataset import load_metadata


DEFAULT_COLUMNS_TO_SUMMARIZE = (
    "perturbation_id",
    "perturbation",
    "compound_id",
    "compound_name",
    "mechanism_of_action",
    "target",
    "pathway",
    "perturbation_type",
    "batch",
    "cell_type",
    "cell_line",
    "experiment",
    "plate",
    "well",
    "site",
)


def summarize_metadata(
    metadata: pd.DataFrame,
    columns_to_summarize: tuple[str, ...] = DEFAULT_COLUMNS_TO_SUMMARIZE,
) -> pd.DataFrame:
    """Build a compact summary table for Cell Painting metadata."""
    rows = [
        {"metric": "rows", "value": len(metadata)},
        {"metric": "columns", "value": len(metadata.columns)},
    ]

    for column in columns_to_summarize:
        if column in metadata.columns:
            rows.append({"metric": f"unique_{column}", "value": metadata[column].nunique()})
            rows.append(
                {
                    "metric": f"missing_{column}",
                    "value": int(metadata[column].isna().sum()),
                }
            )

    return pd.DataFrame(rows)


def top_value_counts(
    metadata: pd.DataFrame,
    column: str,
    top_n: int,
) -> pd.DataFrame:
    """Return top value counts for one metadata column."""
    if column not in metadata.columns:
        return pd.DataFrame(columns=["column", "value", "count"])

    counts = metadata[column].value_counts(dropna=False).head(top_n).reset_index()
    counts.columns = ["value", "count"]
    counts.insert(0, "column", column)
    return counts


def inspect_metadata(
    metadata_path: str | Path,
    output_path: str | Path | None = None,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Load metadata and produce summary and count tables."""
    metadata = load_metadata(metadata_path)
    summary = summarize_metadata(metadata)
    counts = pd.concat(
        [
            top_value_counts(metadata, column, top_n)
            for column in DEFAULT_COLUMNS_TO_SUMMARIZE
        ],
        ignore_index=True,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        counts_path = output_path.with_name(f"{output_path.stem}_top_counts.csv")
        counts.to_csv(counts_path, index=False)

    return {"summary": summary, "top_counts": counts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Cell Painting metadata for phenotypic profiling and retrieval analysis."
    )
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV or Parquet.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the summary CSV.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top values to save per metadata column.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tables = inspect_metadata(args.metadata, args.output, args.top_n)
    print(tables["summary"].to_string(index=False))


if __name__ == "__main__":
    main()

