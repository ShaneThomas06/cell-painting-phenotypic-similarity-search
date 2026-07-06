import numpy as np
import pandas as pd

from cell_painting_profiling.embeddings.retrieval import (
    build_neighbor_table,
    shared_annotation_at_k,
    top_k_retrieval_matches,
)


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


def test_build_neighbor_table_returns_ranked_neighbors():
    query_embeddings = np.array([[1.0, 0.0]])
    reference_embeddings = np.array([[0.9, 0.1], [0.1, 0.9]])
    query_metadata = pd.DataFrame({"perturbation_id": ["query_a"]})
    reference_metadata = pd.DataFrame({"perturbation_id": ["neighbor_a", "neighbor_b"]})

    neighbors = build_neighbor_table(
        query_embeddings,
        reference_embeddings,
        query_metadata,
        reference_metadata,
        query_id_column="perturbation_id",
        reference_id_column="perturbation_id",
        k=2,
    )

    assert list(neighbors["neighbor_id"]) == ["neighbor_a", "neighbor_b"]
    assert list(neighbors["rank"]) == [1, 2]


def test_shared_annotation_at_k_detects_mechanism_recovery():
    neighbor_table = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2"],
            "neighbor_id": ["n1", "n2", "n3"],
            "rank": [1, 2, 1],
            "similarity": [0.9, 0.8, 0.7],
        }
    )
    query_annotations = pd.Series({"q1": "HDAC inhibitor", "q2": "EGFR inhibitor"})
    reference_annotations = pd.Series(
        {"n1": "HDAC inhibitor", "n2": "other", "n3": "other"}
    )

    score = shared_annotation_at_k(
        neighbor_table,
        query_annotations,
        reference_annotations,
        k=1,
    )

    assert score == 0.5
