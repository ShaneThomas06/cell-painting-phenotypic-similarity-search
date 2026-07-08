from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EXPERIMENTS = [
    {
        "experiment_id": "random_12site_finetuned",
        "display_name": "Random ResNet18 fine-tuned",
        "representation": "random initialization",
        "cnn_training": "fine-tuned",
        "training_metrics": "baseline_resnet18_12site_3epoch_metrics.json",
        "retrieval_table": "baseline_12site_trained_resnet18_nearest_neighbors.csv",
        "linear_probe": None,
    },
    {
        "experiment_id": "pretrained_augmented_finetuned",
        "display_name": "Pretrained ResNet18 fine-tuned",
        "representation": "ImageNet pretrained plus augmentation",
        "cnn_training": "fine-tuned",
        "training_metrics": "baseline_resnet18_12site_pretrained_augmented_3epoch_metrics.json",
        "retrieval_table": "baseline_12site_pretrained_augmented_resnet18_nearest_neighbors.csv",
        "linear_probe": None,
    },
    {
        "experiment_id": "frozen_pretrained",
        "display_name": "Frozen pretrained ResNet18",
        "representation": "ImageNet pretrained frozen backbone",
        "cnn_training": "frozen",
        "training_metrics": None,
        "retrieval_table": "baseline_12site_frozen_pretrained_resnet18_nearest_neighbors.csv",
        "linear_probe": "baseline_12site_frozen_pretrained_linear_probe.json",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_retrieval(path: Path) -> dict[str, float]:
    neighbors = pd.read_csv(path)
    return {
        "top1_shared_mechanism": float(neighbors.loc[neighbors["rank"] == 1, "shared_mechanism"].mean()),
        "top3_row_shared_mechanism": float(neighbors["shared_mechanism"].mean()),
        "query_has_shared_mechanism_top3": float(
            neighbors.groupby("query_id")["shared_mechanism"].any().mean()
        ),
    }


def summarize_training(path: Path | None) -> dict[str, float | None]:
    if path is None:
        return {
            "validation_accuracy": None,
            "validation_balanced_accuracy": None,
            "validation_macro_f1": None,
            "final_train_loss": None,
            "final_validation_loss": None,
        }
    metrics = read_json(path)
    final_epoch = metrics["history"][-1]
    return {
        "validation_accuracy": float(final_epoch["val"]["accuracy"]),
        "validation_balanced_accuracy": float(final_epoch["val"]["balanced_accuracy"]),
        "validation_macro_f1": float(final_epoch["val"]["macro_f1"]),
        "final_train_loss": float(final_epoch["train"]["loss"]),
        "final_validation_loss": float(final_epoch["val"]["loss"]),
    }


def summarize_linear_probe(path: Path | None) -> dict[str, float | None]:
    if path is None:
        return {
            "linear_probe_image_accuracy": None,
            "linear_probe_compound_accuracy": None,
            "linear_probe_compound_macro_f1": None,
        }
    metrics = read_json(path)
    return {
        "linear_probe_image_accuracy": float(metrics["image_level_metrics"]["accuracy"]),
        "linear_probe_compound_accuracy": float(metrics["compound_level_metrics"]["accuracy"]),
        "linear_probe_compound_macro_f1": float(metrics["compound_level_metrics"]["macro_f1"]),
    }


def build_comparison_table(tables_dir: str | Path) -> pd.DataFrame:
    tables_dir = Path(tables_dir)
    rows = []
    for experiment in EXPERIMENTS:
        row = {
            "experiment_id": experiment["experiment_id"],
            "display_name": experiment["display_name"],
            "representation": experiment["representation"],
            "cnn_training": experiment["cnn_training"],
        }
        training_path = (
            tables_dir / experiment["training_metrics"]
            if experiment["training_metrics"] is not None
            else None
        )
        linear_probe_path = (
            tables_dir / experiment["linear_probe"]
            if experiment["linear_probe"] is not None
            else None
        )
        row.update(summarize_training(training_path))
        row.update(summarize_retrieval(tables_dir / experiment["retrieval_table"]))
        row.update(summarize_linear_probe(linear_probe_path))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_retrieval_comparison(comparison: pd.DataFrame, output_path: str | Path) -> None:
    plot_frame = comparison.melt(
        id_vars=["display_name"],
        value_vars=["top1_shared_mechanism", "query_has_shared_mechanism_top3"],
        var_name="metric",
        value_name="score",
    )
    metric_labels = {
        "top1_shared_mechanism": "Top-1 shared MOA",
        "query_has_shared_mechanism_top3": "Query has shared MOA in top 3",
    }
    plot_frame["metric"] = plot_frame["metric"].map(metric_labels)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(
        data=plot_frame,
        x="score",
        y="display_name",
        hue="metric",
        ax=ax,
    )
    ax.set_xlabel("Retrieval score")
    ax.set_ylabel("")
    ax.set_xlim(0, max(0.35, float(plot_frame["score"].max()) + 0.05))
    ax.set_title("Mechanism-aware retrieval across representation strategies")
    ax.legend(title="Metric", loc="lower right")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Cell Painting representation experiments.")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--output-figure", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison = build_comparison_table(args.tables_dir)
    output_table = Path(args.output_table)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_table, index=False)
    plot_retrieval_comparison(comparison, args.output_figure)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

