from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cell_painting_profiling.data.multichannel_dataset import (
    DEFAULT_CHANNEL_ORDER,
    MultiChannelCellPaintingDataset,
)
from cell_painting_profiling.models.encoders import build_resnet18_classifier
from cell_painting_profiling.training.forward_smoke_test import collate_batch


class ResNetEmbeddingModel(nn.Module):
    """ResNet18 wrapper that returns penultimate-layer embeddings."""

    def __init__(self, num_input_channels: int = 5) -> None:
        super().__init__()
        classifier = build_resnet18_classifier(
            num_classes=1,
            num_input_channels=num_input_channels,
            pretrained=False,
        )
        self.features = nn.Sequential(*list(classifier.children())[:-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.features(images)
        return torch.flatten(embeddings, start_dim=1)


def extract_image_embeddings(
    manifest: pd.DataFrame,
    image_size: int = 224,
    batch_size: int = 4,
) -> pd.DataFrame:
    """Extract image-level CNN embeddings from a channel-level manifest."""
    dataset = MultiChannelCellPaintingDataset(manifest, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    model = ResNetEmbeddingModel(num_input_channels=len(DEFAULT_CHANNEL_ORDER))
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            embeddings = model(batch["image"]).cpu().numpy()
            for index, image_record_id in enumerate(batch["image_record_id"]):
                row = {
                    "image_record_id": image_record_id,
                    "mechanism_of_action": batch["mechanism_of_action"][index],
                }
                for embedding_index, value in enumerate(embeddings[index]):
                    row[f"embedding_{embedding_index:04d}"] = float(value)
                rows.append(row)

    metadata_columns = [
        "image_record_id",
        "perturbation_id",
        "compound_name",
        "mechanism_of_action",
        "well",
        "site",
    ]
    image_metadata = manifest[metadata_columns].drop_duplicates("image_record_id")
    embeddings = pd.DataFrame(rows)
    return image_metadata.merge(embeddings, on=["image_record_id", "mechanism_of_action"], how="left")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract CNN embeddings from Cell Painting images.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = pd.read_csv(args.manifest)
    embeddings = extract_image_embeddings(
        manifest,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(output, index=False)
    print(f"Wrote {len(embeddings)} image embeddings to {output}")


if __name__ == "__main__":
    main()
