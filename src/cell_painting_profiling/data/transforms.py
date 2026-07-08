from __future__ import annotations

import json
from pathlib import Path

import torch


DEFAULT_CHANNEL_MEAN = (0.5, 0.5, 0.5, 0.5, 0.5)
DEFAULT_CHANNEL_STD = (0.25, 0.25, 0.25, 0.25, 0.25)


def normalize_channel_stack(
    image: torch.Tensor,
    mean: tuple[float, ...] = DEFAULT_CHANNEL_MEAN,
    std: tuple[float, ...] = DEFAULT_CHANNEL_STD,
) -> torch.Tensor:
    """Normalize a channel-first microscopy tensor."""
    mean_tensor = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    if image.shape[0] != len(mean):
        raise ValueError("Channel count does not match normalization constants")
    return (image - mean_tensor) / std_tensor.clamp_min(1e-8)


def load_channel_stats(
    stats_path: str | Path,
    channel_order: tuple[str, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Load channel mean/std values from a statistics JSON file."""
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    channels = stats.get("channels", {})
    missing = set(channel_order).difference(channels)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Channel stats are missing channels: {missing_text}")
    mean = tuple(float(channels[channel]["mean"]) for channel in channel_order)
    std = tuple(float(channels[channel]["std"]) for channel in channel_order)
    return mean, std


def augment_channel_stack(image: torch.Tensor) -> torch.Tensor:
    """Apply light orientation-preserving augmentation to a channel-first tensor."""
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-1,))
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-2,))
    rotations = int(torch.randint(0, 4, ()).item())
    if rotations:
        image = torch.rot90(image, k=rotations, dims=(-2, -1))
    return image


class ChannelStackTransform:
    """Tensor transform for 5-channel Cell Painting image stacks."""

    def __init__(
        self,
        train: bool = False,
        normalize: bool = True,
        mean: tuple[float, ...] = DEFAULT_CHANNEL_MEAN,
        std: tuple[float, ...] = DEFAULT_CHANNEL_STD,
    ) -> None:
        self.train = train
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.train:
            image = augment_channel_stack(image)
        if self.normalize:
            image = normalize_channel_stack(image, mean=self.mean, std=self.std)
        return image
