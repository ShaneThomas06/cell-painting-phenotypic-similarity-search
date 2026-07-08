import json

import pandas as pd

from cell_painting_profiling.training.linear_probe import run_linear_probe


def test_run_linear_probe_outputs_image_and_compound_metrics(tmp_path):
    rows = []
    for mechanism_index, mechanism in enumerate(["m1", "m2"]):
        for compound_index, compound_suffix in enumerate(["a", "b"]):
            compound_id = f"{mechanism}_{compound_suffix}"
            base = mechanism_index * 10 + compound_index
            for site in range(3):
                rows.append(
                    {
                        "image_record_id": f"{compound_id}_{site}",
                        "perturbation_id": compound_id,
                        "compound_name": compound_id.upper(),
                        "mechanism_of_action": mechanism,
                        "embedding_0000": float(base + site * 0.1),
                        "embedding_0001": float(mechanism_index),
                    }
                )
    embeddings_path = tmp_path / "embeddings.csv"
    output_path = tmp_path / "linear_probe.json"
    pd.DataFrame(rows).to_csv(embeddings_path, index=False)

    result = run_linear_probe(
        embeddings_path=embeddings_path,
        output_path=output_path,
        max_iter=200,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["split_counts"]["train_compounds"] == 2
    assert result["split_counts"]["val_compounds"] == 2
    assert "accuracy" in result["image_level_metrics"]
    assert "accuracy" in result["compound_level_metrics"]
    assert len(saved["compound_predictions"]) == 2
