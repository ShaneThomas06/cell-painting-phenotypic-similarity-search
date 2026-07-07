from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_CHANNEL_ORDER = ("rna", "mito", "agp", "er", "dna")


def load_channel_image(path: str | Path) -> torch.Tensor:
    """Load a single-channel TIFF image as a float tensor scaled to [0, 1]."""
    with Image.open(path) as image:
        array = np.array(image)

    tensor = torch.as_tensor(array, dtype=torch.float32)
    if np.issubdtype(array.dtype, np.integer):
        max_value = float(np.iinfo(array.dtype).max)
    else:
        max_value = float(tensor.max().item()) if tensor.numel() else 1.0
    if max_value <= 0:
        max_value = 1.0
    return tensor / max_value


def resize_channel_stack(image: torch.Tensor, image_size: int | None) -> torch.Tensor:
    """Resize a channel-first image tensor while preserving channel count."""
    if image_size is None:
        return image
    if image.shape[-2:] == (image_size, image_size):
        return image
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


class MultiChannelCellPaintingDataset(Dataset):
    """Load one Cell Painting image site as a stacked multi-channel tensor."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        channel_order: Sequence[str] = DEFAULT_CHANNEL_ORDER,
        image_size: int | None = 224,
    ) -> None:
        required = {
            "image_record_id",
            "channel",
            "local_path",
            "perturbation_id",
            "compound_name",
            "mechanism_of_action",
        }
        missing = required.difference(manifest.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Manifest is missing required columns: {missing_text}")

        self.manifest = manifest.copy()
        self.channel_order = tuple(channel_order)
        self.image_size = image_size
        self.records = self._build_records(self.manifest)

    def _build_records(self, manifest: pd.DataFrame) -> list[dict]:
        records = []
        for image_record_id, group in manifest.groupby("image_record_id", sort=True):
            channels = dict(zip(group["channel"], group["local_path"], strict=False))
            missing_channels = set(self.channel_order).difference(channels)
            if missing_channels:
                missing_text = ", ".join(sorted(missing_channels))
                raise ValueError(f"{image_record_id} is missing channels: {missing_text}")

            first = group.iloc[0]
            records.append(
                {
                    "image_record_id": image_record_id,
                    "channel_paths": [channels[channel] for channel in self.channel_order],
                    "perturbation_id": first["perturbation_id"],
                    "compound_name": first["compound_name"],
                    "mechanism_of_action": first["mechanism_of_action"],
                    "well": first.get("well"),
                    "site": first.get("site"),
                }
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        channels = [load_channel_image(path) for path in record["channel_paths"]]
        image = torch.stack(channels, dim=0)
        image = resize_channel_stack(image, self.image_size)
        return {
            "image": image,
            "image_record_id": record["image_record_id"],
            "perturbation_id": record["perturbation_id"],
            "compound_name": record["compound_name"],
            "mechanism_of_action": record["mechanism_of_action"],
            "well": record["well"],
            "site": record["site"],
        }

