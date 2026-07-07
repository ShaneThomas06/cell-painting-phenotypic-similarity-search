import pandas as pd

from cell_painting_profiling.embeddings.run_retrieval import run_compound_retrieval


def test_run_compound_retrieval_excludes_self_neighbors():
    fingerprints = pd.DataFrame(
        {
            "perturbation_id": ["a", "b", "c"],
            "compound_name": ["A", "B", "C"],
            "mechanism_of_action": ["m1", "m1", "m2"],
            "embedding_0000": [1.0, 0.9, 0.0],
            "embedding_0001": [0.0, 0.1, 1.0],
        }
    )

    neighbors = run_compound_retrieval(fingerprints, k=1)

    assert len(neighbors) == 3
    assert not (neighbors["query_id"] == neighbors["neighbor_id"]).any()
    assert set(neighbors["rank"]) == {1}
    assert "shared_mechanism" in neighbors.columns

