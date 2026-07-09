import pandas as pd

from cell_painting_profiling.analysis.visualize_results import select_retrieval_examples


def test_select_retrieval_examples_returns_success_and_failure_cases():
    neighbors = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q2"],
            "neighbor_id": ["n1", "n2", "n3", "n4"],
            "rank": [1, 2, 1, 2],
            "shared_mechanism": [True, False, False, True],
            "query_compound_name": ["Q1", "Q1", "Q2", "Q2"],
            "query_mechanism_of_action": ["m1", "m1", "m2", "m2"],
            "neighbor_compound_name": ["N1", "N2", "N3", "N4"],
            "neighbor_mechanism_of_action": ["m1", "m3", "m4", "m2"],
            "similarity": [0.9, 0.8, 0.7, 0.6],
        }
    )

    examples = select_retrieval_examples(neighbors)

    assert [example["label"] for example in examples] == [
        "Shared mechanism top-1",
        "Different mechanism top-1",
    ]
    assert examples[0]["query_id"] == "q1"
    assert examples[1]["query_id"] == "q2"
