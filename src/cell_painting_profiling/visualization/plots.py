import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_class_counts(metadata: pd.DataFrame, column: str, top_n: int = 30):
    """Plot the most common labels in a metadata column."""
    counts = metadata[column].value_counts().head(top_n).reset_index()
    counts.columns = [column, "count"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=counts, x="count", y=column, ax=ax)
    ax.set_title(f"Top {top_n} values for {column}")
    ax.set_xlabel("Count")
    ax.set_ylabel(column)
    fig.tight_layout()
    return fig, ax
