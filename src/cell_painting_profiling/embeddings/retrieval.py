import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def top_k_retrieval_matches(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    query_labels: np.ndarray,
    reference_labels: np.ndarray,
    k: int = 5,
) -> float:
    """Return the fraction of queries with a matching label in top-k neighbors."""
    similarities = cosine_similarity(query_embeddings, reference_embeddings)
    top_indices = np.argsort(-similarities, axis=1)[:, :k]
    matches = []
    for query_index, neighbors in enumerate(top_indices):
        neighbor_labels = reference_labels[neighbors]
        matches.append(query_labels[query_index] in neighbor_labels)
    return float(np.mean(matches))


def build_neighbor_table(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    query_metadata: pd.DataFrame,
    reference_metadata: pd.DataFrame,
    query_id_column: str,
    reference_id_column: str,
    k: int = 10,
) -> pd.DataFrame:
    """Build a nearest-neighbor table for phenotypic similarity search."""
    similarities = cosine_similarity(query_embeddings, reference_embeddings)
    top_indices = np.argsort(-similarities, axis=1)[:, :k]

    rows = []
    for query_index, neighbors in enumerate(top_indices):
        query_id = query_metadata.iloc[query_index][query_id_column]
        for rank, reference_index in enumerate(neighbors, start=1):
            reference_id = reference_metadata.iloc[reference_index][reference_id_column]
            rows.append(
                {
                    "query_id": query_id,
                    "neighbor_id": reference_id,
                    "rank": rank,
                    "similarity": float(similarities[query_index, reference_index]),
                }
            )

    return pd.DataFrame(rows)


def shared_annotation_at_k(
    neighbor_table: pd.DataFrame,
    query_annotations: pd.Series,
    reference_annotations: pd.Series,
    k: int,
) -> float:
    """Measure whether each query has at least one neighbor sharing an annotation."""
    if neighbor_table.empty:
        return 0.0

    top_neighbors = neighbor_table.loc[neighbor_table["rank"] <= k]
    matches = []
    for query_id, query_neighbors in top_neighbors.groupby("query_id"):
        if query_id not in query_annotations.index:
            continue
        query_value = query_annotations.loc[query_id]
        neighbor_values = reference_annotations.reindex(query_neighbors["neighbor_id"]).dropna()
        matches.append(query_value in set(neighbor_values))

    if not matches:
        return 0.0
    return float(np.mean(matches))
