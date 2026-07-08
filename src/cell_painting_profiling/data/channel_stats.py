from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from cell_painting_profiling.data.multichannel_dataset import (
    DEFAULT_CHANNEL_ORDER,
    load_channel_image,
    resize_channel_stack,
)


def compute_channel_stats(
    manifest: pd.DataFrame,
    channel_order: tuple[str, ...] = DEFAULT_CHANNEL_ORDER,
    image_size: int | None = None,
) -> dict:
    """Compute per-channel mean and standard deviation from local image files."""
    required = {"channel", "local_path"}
    missing = required.difference(manifest.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Manifest is missing required columns: {missing_text}")

    channel_set = set(channel_order)
    sums = {channel: 0.0 for channel in channel_order}
    squared_sums = {channel: 0.0 for channel in channel_order}
    pixel_counts = {channel: 0 for channel in channel_order}
    file_counts = {channel: 0 for channel in channel_order}

    for _, row in manifest.iterrows():
        channel = row["channel"]
        if channel not in channel_set:
            continue
        image = load_channel_image(row["local_path"])
        if image_size is not None:
            image = resize_channel_stack(image.unsqueeze(0), image_size).squeeze(0)
        image = image.to(dtype=torch.float64)
        sums[channel] += float(image.sum().item())
        squared_sums[channel] += float((image * image).sum().item())
        pixel_counts[channel] += int(image.numel())
        file_counts[channel] += 1

    channels = {}
    for channel in channel_order:
        if pixel_counts[channel] == 0:
            raise ValueError(f"No pixels found for channel: {channel}")
        mean = sums[channel] / pixel_counts[channel]
        variance = (squared_sums[channel] / pixel_counts[channel]) - (mean * mean)
        std = max(variance, 0.0) ** 0.5
        channels[channel] = {
            "mean": mean,
            "std": std,
            "num_pixels": pixel_counts[channel],
            "num_files": file_counts[channel],
        }

    return {
        "image_size": image_size,
        "channel_order": list(channel_order),
        "channels": channels,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Cell Painting channel statistics.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = pd.read_csv(args.manifest)
    stats = compute_channel_stats(manifest, image_size=args.image_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
