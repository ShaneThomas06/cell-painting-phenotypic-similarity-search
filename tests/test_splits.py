import pandas as pd

from cell_painting_profiling.data.splits import (
    filter_balanced_subset,
    split_by_held_out_batches,
)


def test_split_by_held_out_batches():
    metadata = pd.DataFrame(
        {
            "batch": ["a", "a", "b", "c"],
            "perturbation": ["p1", "p2", "p1", "p2"],
        }
    )

    train, test = split_by_held_out_batches(metadata, held_out_batches=["c"])

    assert set(train["batch"]) == {"a", "b"}
    assert set(test["batch"]) == {"c"}


def test_filter_balanced_subset_respects_min_count():
    metadata = pd.DataFrame(
        {
            "cell_type": ["x", "x", "x", "x", "x"],
            "perturbation": ["p1", "p1", "p2", "p3", "p3"],
        }
    )

    subset = filter_balanced_subset(metadata, min_images_per_perturbation=2)

    assert set(subset["perturbation"]) == {"p1", "p3"}
