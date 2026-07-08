from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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

    def __init__(
        self,
        num_input_channels: int = 5,
        classifier: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if classifier is None:
            classifier = build_resnet18_classifier(
                num_classes=1,
                num_input_channels=num_input_channels,
                pretrained=False,
            )
        self.features = nn.Sequential(*list(classifier.children())[:-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.features(images)
        return torch.flatten(embeddings, start_dim=1)


def load_trained_embedding_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ResNetEmbeddingModel, dict[str, Any]]:
    """Load a trained classifier checkpoint as an embedding model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_map = checkpoint.get("label_map")
    if not label_map:
        raise ValueError("Checkpoint is missing label_map")

    channel_order = checkpoint.get("channel_order", list(DEFAULT_CHANNEL_ORDER))
    classifier = build_resnet18_classifier(
        num_classes=len(label_map),
        num_input_channels=len(channel_order),
        pretrained=False,
    )
    classifier.load_state_dict(checkpoint["model_state_dict"])
    model = ResNetEmbeddingModel(
        num_input_channels=len(channel_order),
        classifier=classifier,
    ).to(device)
    model.eval()
    return model, checkpoint


def extract_image_embeddings(
    manifest: pd.DataFrame,
    image_size: int = 224,
    batch_size: int = 4,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    """Extract image-level CNN embeddings from a channel-level manifest."""
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = MultiChannelCellPaintingDataset(manifest, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    if checkpoint_path is None:
        model = ResNetEmbeddingModel(num_input_channels=len(DEFAULT_CHANNEL_ORDER)).to(torch_device)
        checkpoint_metadata: dict[str, Any] = {}
    else:
        model, checkpoint_metadata = load_trained_embedding_model(checkpoint_path, torch_device)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(torch_device)
            embeddings = model(images).cpu().numpy()
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
    merged = image_metadata.merge(
        embeddings,
        on=["image_record_id", "mechanism_of_action"],
        how="left",
    )
    if checkpoint_path is not None:
        merged.attrs["checkpoint_path"] = str(checkpoint_path)
        merged.attrs["checkpoint_image_size"] = checkpoint_metadata.get("image_size")
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract CNN embeddings from Cell Painting images.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint", help="Optional trained classifier checkpoint.")
    parser.add_argument("--device", help="Optional torch device, such as cpu or cuda.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = pd.read_csv(args.manifest)
    embeddings = extract_image_embeddings(
        manifest,
        image_size=args.image_size,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(output, index=False)
    print(f"Wrote {len(embeddings)} image embeddings to {output}")


if __name__ == "__main__":
    main()
