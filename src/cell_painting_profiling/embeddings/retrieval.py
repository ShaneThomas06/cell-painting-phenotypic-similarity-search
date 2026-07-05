import numpy as np
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
