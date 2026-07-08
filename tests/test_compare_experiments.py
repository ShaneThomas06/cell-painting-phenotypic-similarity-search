import json

import pandas as pd

from cell_painting_profiling.analysis.compare_experiments import (
    build_comparison_table,
    summarize_retrieval,
)


def write_training_metrics(path, accuracy=0.25):
    path.write_text(
        json.dumps(
            {
                "history": [
                    {
                        "epoch": 1,
                        "train": {"loss": 1.0},
                        "val": {
                            "loss": 2.0,
                            "accuracy": accuracy,
                            "balanced_accuracy": accuracy,
                            "macro_f1": 0.2,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def write_retrieval_table(path):
    pd.DataFrame(
        {
            "query_id": ["a", "a", "b", "b"],
            "neighbor_id": ["x", "y", "z", "w"],
            "rank": [1, 2, 1, 2],
            "shared_mechanism": [True, False, False, True],
        }
    ).to_csv(path, index=False)


def write_linear_probe(path):
    path.write_text(
        json.dumps(
            {
                "image_level_metrics": {"accuracy": 0.3},
                "compound_level_metrics": {"accuracy": 0.2, "macro_f1": 0.1},
            }
        ),
        encoding="utf-8",
    )


def test_summarize_retrieval_computes_topk_metrics(tmp_path):
    path = tmp_path / "retrieval.csv"
    write_retrieval_table(path)

    metrics = summarize_retrieval(path)

    assert metrics["top1_shared_mechanism"] == 0.5
    assert metrics["top3_row_shared_mechanism"] == 0.5
    assert metrics["query_has_shared_mechanism_top3"] == 1.0


def test_build_comparison_table_reads_expected_artifacts(tmp_path):
    write_training_metrics(tmp_path / "baseline_resnet18_12site_3epoch_metrics.json")
    write_training_metrics(tmp_path / "baseline_resnet18_12site_pretrained_augmented_3epoch_metrics.json")
    write_retrieval_table(tmp_path / "baseline_12site_trained_resnet18_nearest_neighbors.csv")
    write_retrieval_table(tmp_path / "baseline_12site_pretrained_augmented_resnet18_nearest_neighbors.csv")
    write_retrieval_table(tmp_path / "baseline_12site_frozen_pretrained_resnet18_nearest_neighbors.csv")
    write_linear_probe(tmp_path / "baseline_12site_frozen_pretrained_linear_probe.json")

    comparison = build_comparison_table(tmp_path)

    assert comparison.shape[0] == 3
    assert set(comparison["experiment_id"]) == {
        "random_12site_finetuned",
        "pretrained_augmented_finetuned",
        "frozen_pretrained",
    }
    frozen = comparison.loc[comparison["experiment_id"] == "frozen_pretrained"].iloc[0]
    assert frozen["linear_probe_image_accuracy"] == 0.3
