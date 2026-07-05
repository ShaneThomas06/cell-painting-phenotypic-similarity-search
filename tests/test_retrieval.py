import numpy as np

from cell_painting_profiling.embeddings.retrieval import top_k_retrieval_matches


def test_top_k_retrieval_matches():
    query_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    reference_embeddings = np.array([[0.9, 0.1], [0.1, 0.9]])
    query_labels = np.array(["a", "b"])
    reference_labels = np.array(["a", "b"])

    score = top_k_retrieval_matches(
        query_embeddings,
        reference_embeddings,
        query_labels,
        reference_labels,
        k=1,
    )

    assert score == 1.0
