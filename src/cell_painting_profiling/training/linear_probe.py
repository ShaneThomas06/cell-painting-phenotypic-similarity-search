from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cell_painting_profiling.embeddings.aggregate import get_embedding_columns
from cell_painting_profiling.training.forward_smoke_test import build_mechanism_label_map
from cell_painting_profiling.training.metrics import classification_metrics
from cell_painting_profiling.training.train import assign_compound_holdout_split


def build_feature_matrix(frame: pd.DataFrame, embedding_columns: list[str]) -> np.ndarray:
    return frame[embedding_columns].to_numpy(dtype=np.float64)


def predict_compound_labels(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    class_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate image-level probabilities into compound-level predictions."""
    probability_frame = pd.DataFrame(probabilities, columns=class_labels)
    probability_frame["perturbation_id"] = frame["perturbation_id"].to_numpy()
    probability_frame["mechanism_of_action"] = frame["mechanism_of_action"].to_numpy()
    mean_probabilities = probability_frame.groupby("perturbation_id")[list(class_labels)].mean()
    metadata = frame[["perturbation_id", "mechanism_of_action"]].drop_duplicates("perturbation_id")
    metadata = metadata.set_index("perturbation_id").loc[mean_probabilities.index]
    y_true = metadata["mechanism_of_action"].to_numpy()
    y_pred = mean_probabilities.idxmax(axis=1).to_numpy()
    return mean_probabilities.index.to_numpy(), y_true, y_pred


def run_linear_probe(
    embeddings_path: str | Path,
    output_path: str | Path,
    max_iter: int = 1000,
    c_value: float = 1.0,
    val_compounds_per_mechanism: int = 1,
) -> dict[str, Any]:
    """Train a logistic-regression probe on frozen image embeddings."""
    embeddings = pd.read_csv(embeddings_path)
    embedding_columns = get_embedding_columns(embeddings)
    if not embedding_columns:
        raise ValueError("Embeddings table has no embedding_* columns")

    split = assign_compound_holdout_split(
        embeddings,
        val_compounds_per_mechanism=val_compounds_per_mechanism,
    )
    train = split.loc[split["split"] == "train"].reset_index(drop=True)
    val = split.loc[split["split"] == "val"].reset_index(drop=True)
    label_map = build_mechanism_label_map(split)
    inverse_label_map = {index: label for label, index in label_map.items()}

    x_train = build_feature_matrix(train, embedding_columns)
    x_val = build_feature_matrix(val, embedding_columns)
    y_train = train["mechanism_of_action"].map(label_map).to_numpy()
    y_val = val["mechanism_of_action"].map(label_map).to_numpy()

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    classifier.fit(x_train, y_train)

    val_pred = classifier.predict(x_val)
    image_metrics = classification_metrics(y_val, val_pred)
    val_probabilities = classifier.predict_proba(x_val)
    class_labels = np.array([inverse_label_map[index] for index in classifier.classes_])
    compound_ids, compound_true, compound_pred = predict_compound_labels(
        val,
        val_probabilities,
        class_labels,
    )
    compound_metrics = classification_metrics(compound_true, compound_pred)

    result: dict[str, Any] = {
        "embeddings_path": str(embeddings_path),
        "num_embedding_features": len(embedding_columns),
        "max_iter": max_iter,
        "c_value": c_value,
        "label_map": label_map,
        "split_counts": {
            "train_image_records": int(train["image_record_id"].nunique()),
            "val_image_records": int(val["image_record_id"].nunique()),
            "train_compounds": int(train["perturbation_id"].nunique()),
            "val_compounds": int(val["perturbation_id"].nunique()),
        },
        "image_level_metrics": image_metrics,
        "compound_level_metrics": compound_metrics,
        "compound_predictions": [
            {
                "perturbation_id": perturbation_id,
                "true_mechanism": true_label,
                "predicted_mechanism": predicted_label,
            }
            for perturbation_id, true_label, predicted_label in zip(
                compound_ids,
                compound_true,
                compound_pred,
                strict=False,
            )
        ],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a linear probe on frozen embeddings.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--val-compounds-per-mechanism", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_linear_probe(
        embeddings_path=args.embeddings,
        output_path=args.output,
        max_iter=args.max_iter,
        c_value=args.c_value,
        val_compounds_per_mechanism=args.val_compounds_per_mechanism,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

