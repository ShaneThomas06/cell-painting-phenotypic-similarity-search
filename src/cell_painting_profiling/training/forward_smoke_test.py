from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cell_painting_profiling.data.multichannel_dataset import (
    DEFAULT_CHANNEL_ORDER,
    MultiChannelCellPaintingDataset,
)
from cell_painting_profiling.models.encoders import build_resnet18_classifier


def build_mechanism_label_map(manifest: pd.DataFrame) -> dict[str, int]:
    mechanisms = sorted(manifest["mechanism_of_action"].dropna().unique())
    return {mechanism: index for index, mechanism in enumerate(mechanisms)}


def collate_batch(batch: list[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch], dim=0)
    return {
        "image": images,
        "image_record_id": [item["image_record_id"] for item in batch],
        "mechanism_of_action": [item["mechanism_of_action"] for item in batch],
    }


def run_forward_smoke_test(
    manifest_path: str | Path,
    output_path: str | Path,
    image_size: int = 224,
    batch_size: int = 4,
) -> dict:
    manifest = pd.read_csv(manifest_path)
    label_map = build_mechanism_label_map(manifest)
    dataset = MultiChannelCellPaintingDataset(manifest, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = build_resnet18_classifier(
        num_classes=len(label_map),
        num_input_channels=len(DEFAULT_CHANNEL_ORDER),
        pretrained=False,
    )
    model.eval()

    batch = next(iter(loader))
    with torch.no_grad():
        logits = model(batch["image"])

    result = {
        "manifest_path": str(manifest_path),
        "num_image_records": len(dataset),
        "num_channel_rows": len(manifest),
        "channel_order": list(DEFAULT_CHANNEL_ORDER),
        "image_size": image_size,
        "batch_size": batch["image"].shape[0],
        "input_shape": list(batch["image"].shape),
        "num_mechanism_classes": len(label_map),
        "output_shape": list(logits.shape),
        "mechanism_label_map": label_map,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a CNN forward-pass smoke test on a Cell Painting manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_forward_smoke_test(
        manifest_path=args.manifest,
        output_path=args.output,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
