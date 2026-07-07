import pandas as pd

from cell_painting_profiling.embeddings.aggregate import aggregate_compound_fingerprints


def test_aggregate_compound_fingerprints_averages_image_embeddings():
    embeddings = pd.DataFrame(
        {
            "image_record_id": ["a_1", "a_2", "b_1"],
            "perturbation_id": ["a", "a", "b"],
            "compound_name": ["A", "A", "B"],
            "mechanism_of_action": ["m1", "m1", "m2"],
            "embedding_0000": [1.0, 3.0, 10.0],
            "embedding_0001": [2.0, 4.0, 20.0],
        }
    )

    fingerprints = aggregate_compound_fingerprints(embeddings)
    a = fingerprints.loc[fingerprints["perturbation_id"] == "a"].iloc[0]

    assert a["embedding_0000"] == 2.0
    assert a["embedding_0001"] == 3.0
    assert a["num_image_records"] == 2
