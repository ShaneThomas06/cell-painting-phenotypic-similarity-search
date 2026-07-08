from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cell_painting_profiling.data.multichannel_dataset import (
    DEFAULT_CHANNEL_ORDER,
    MultiChannelCellPaintingDataset,
)
from cell_painting_profiling.models.encoders import build_resnet18_classifier
from cell_painting_profiling.training.forward_smoke_test import build_mechanism_label_map
from cell_painting_profiling.training.metrics import classification_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assign_compound_holdout_split(
    manifest: pd.DataFrame,
    val_compounds_per_mechanism: int = 1,
) -> pd.DataFrame:
    """Assign train/val splits by holding out compounds within each mechanism."""
    required = {"perturbation_id", "compound_name", "mechanism_of_action"}
    missing = required.difference(manifest.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Manifest is missing required columns: {missing_text}")

    split = manifest.copy()
    split["split"] = "train"
    for mechanism, group in split.groupby("mechanism_of_action", sort=True):
        compounds = (
            group[["perturbation_id", "compound_name"]]
            .drop_duplicates()
            .sort_values(["compound_name", "perturbation_id"])
        )
        if len(compounds) <= val_compounds_per_mechanism:
            raise ValueError(
                f"Mechanism needs more compounds for holdout split: {mechanism}"
            )
        val_ids = set(compounds.tail(val_compounds_per_mechanism)["perturbation_id"])
        split.loc[
            (split["mechanism_of_action"] == mechanism)
            & (split["perturbation_id"].isin(val_ids)),
            "split",
        ] = "val"
    return split


def collate_labeled_batch(batch: list[dict], label_map: dict[str, int]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.tensor(
        [label_map[item["mechanism_of_action"]] for item in batch],
        dtype=torch.long,
    )
    return {
        "image": images,
        "label": labels,
        "image_record_id": [item["image_record_id"] for item in batch],
        "mechanism_of_action": [item["mechanism_of_action"] for item in batch],
    }


def make_loader(
    manifest: pd.DataFrame,
    label_map: dict[str, int],
    image_size: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = MultiChannelCellPaintingDataset(manifest, image_size=image_size)

    def collate(batch: list[dict]) -> dict[str, Any]:
        return collate_labeled_batch(batch, label_map)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * images.shape[0]
        total_examples += images.shape[0]

    if total_examples == 0:
        return {"loss": 0.0}
    return {"loss": total_loss / total_examples}


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * images.shape[0]
            total_examples += images.shape[0]
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0}

    metrics = classification_metrics(np.asarray(y_true), np.asarray(y_pred))
    return {"loss": total_loss / total_examples, **metrics}


def run_training(
    manifest_path: str | Path,
    metrics_output: str | Path,
    model_output: str | Path,
    image_size: int = 224,
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    val_compounds_per_mechanism: int = 1,
    pretrained: bool = False,
    seed: int = 42,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(manifest_path)
    manifest = assign_compound_holdout_split(
        manifest,
        val_compounds_per_mechanism=val_compounds_per_mechanism,
    )
    label_map = build_mechanism_label_map(manifest)
    train_manifest = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
    val_manifest = manifest.loc[manifest["split"] == "val"].reset_index(drop=True)

    train_loader = make_loader(train_manifest, label_map, image_size, batch_size, shuffle=True)
    val_loader = make_loader(val_manifest, label_map, image_size, batch_size, shuffle=False)

    model = build_resnet18_classifier(
        num_classes=len(label_map),
        num_input_channels=len(DEFAULT_CHANNEL_ORDER),
        pretrained=pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_batches=max_train_batches,
        )
        val_metrics = evaluate_classifier(
            model,
            val_loader,
            criterion,
            device,
            max_batches=max_val_batches,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

    model_output = Path(model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_map": label_map,
            "channel_order": list(DEFAULT_CHANNEL_ORDER),
            "image_size": image_size,
        },
        model_output,
    )

    result: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "model_output": str(model_output),
        "device": str(device),
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "pretrained": pretrained,
        "seed": seed,
        "num_classes": len(label_map),
        "label_map": label_map,
        "split_counts": {
            "train_image_records": int(train_manifest["image_record_id"].nunique()),
            "val_image_records": int(val_manifest["image_record_id"].nunique()),
            "train_compounds": int(train_manifest["perturbation_id"].nunique()),
            "val_compounds": int(val_manifest["perturbation_id"].nunique()),
        },
        "history": history,
    }

    metrics_output = Path(metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a baseline ResNet18 MOA classifier.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-compounds-per-mechanism", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_training(
        manifest_path=args.manifest,
        metrics_output=args.metrics_output,
        model_output=args.model_output,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_compounds_per_mechanism=args.val_compounds_per_mechanism,
        pretrained=args.pretrained,
        seed=args.seed,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
