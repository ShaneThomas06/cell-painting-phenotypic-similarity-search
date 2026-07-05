import pandas as pd


def split_by_held_out_batches(
    metadata: pd.DataFrame,
    held_out_batches: list[str],
    batch_column: str = "batch",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split metadata into train and test sets using held-out batches."""
    if batch_column not in metadata.columns:
        raise ValueError(f"Missing batch column: {batch_column}")

    held_out = set(held_out_batches)
    is_test = metadata[batch_column].isin(held_out)
    train = metadata.loc[~is_test].reset_index(drop=True)
    test = metadata.loc[is_test].reset_index(drop=True)
    return train, test


def filter_balanced_subset(
    metadata: pd.DataFrame,
    perturbation_column: str = "perturbation",
    cell_type_column: str = "cell_type",
    cell_type: str | None = None,
    max_perturbations: int | None = None,
    min_images_per_perturbation: int = 1,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Filter metadata to a manageable perturbation-balanced subset."""
    subset = metadata.copy()
    if cell_type is not None and cell_type_column in subset.columns:
        subset = subset.loc[subset[cell_type_column] == cell_type]

    counts = subset[perturbation_column].value_counts()
    eligible = counts[counts >= min_images_per_perturbation].index
    subset = subset.loc[subset[perturbation_column].isin(eligible)]

    if max_perturbations is not None:
        selected = (
            subset[[perturbation_column]]
            .drop_duplicates()
            .sample(
                n=min(max_perturbations, subset[perturbation_column].nunique()),
                random_state=random_seed,
            )[perturbation_column]
        )
        subset = subset.loc[subset[perturbation_column].isin(selected)]

    return subset.reset_index(drop=True)
