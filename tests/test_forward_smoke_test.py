import pandas as pd

from cell_painting_profiling.training.forward_smoke_test import build_mechanism_label_map


def test_build_mechanism_label_map_is_sorted():
    manifest = pd.DataFrame(
        {
            "mechanism_of_action": [
                "JAK inhibitor",
                "HDAC inhibitor",
                "JAK inhibitor",
            ]
        }
    )

    label_map = build_mechanism_label_map(manifest)

    assert label_map == {"HDAC inhibitor": 0, "JAK inhibitor": 1}
