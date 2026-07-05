import pandas as pd

from cell_painting_profiling.data.inspect_metadata import (
    summarize_metadata,
    top_value_counts,
)


def test_summarize_metadata_counts_core_columns():
    metadata = pd.DataFrame(
        {
            "image_path": ["a.png", "b.png", "c.png"],
            "perturbation": ["p1", "p1", "p2"],
            "batch": ["b1", "b2", "b2"],
            "cell_type": ["HUVEC", "HUVEC", "RPE"],
        }
    )

    summary = summarize_metadata(metadata)
    values = dict(zip(summary["metric"], summary["value"], strict=True))

    assert values["rows"] == 3
    assert values["unique_perturbation"] == 2
    assert values["unique_batch"] == 2
    assert values["unique_cell_type"] == 2


def test_top_value_counts_returns_requested_column():
    metadata = pd.DataFrame({"batch": ["b1", "b1", "b2"]})

    counts = top_value_counts(metadata, "batch", top_n=1)

    assert counts.iloc[0]["column"] == "batch"
    assert counts.iloc[0]["value"] == "b1"
    assert counts.iloc[0]["count"] == 2
