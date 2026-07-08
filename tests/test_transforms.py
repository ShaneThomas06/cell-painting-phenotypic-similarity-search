import torch

from cell_painting_profiling.data.transforms import (
    ChannelStackTransform,
    augment_channel_stack,
    load_channel_stats,
    normalize_channel_stack,
)


def test_normalize_channel_stack_preserves_shape_and_centers_values():
    image = torch.full((5, 4, 4), 0.5)

    normalized = normalize_channel_stack(image)

    assert normalized.shape == image.shape
    assert torch.allclose(normalized, torch.zeros_like(normalized))


def test_augment_channel_stack_preserves_shape():
    image = torch.arange(5 * 4 * 4, dtype=torch.float32).reshape(5, 4, 4)

    augmented = augment_channel_stack(image)

    assert augmented.shape == image.shape
    assert set(augmented.flatten().tolist()) == set(image.flatten().tolist())


def test_channel_stack_transform_can_normalize_without_augmentation():
    transform = ChannelStackTransform(train=False, normalize=True)
    image = torch.full((5, 4, 4), 0.5)

    transformed = transform(image)

    assert torch.allclose(transformed, torch.zeros_like(transformed))


def test_load_channel_stats_returns_ordered_mean_std(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        '{"channels":{"rna":{"mean":0.1,"std":0.2},"mito":{"mean":0.3,"std":0.4}}}',
        encoding="utf-8",
    )

    mean, std = load_channel_stats(stats_path, ("mito", "rna"))

    assert mean == (0.3, 0.1)
    assert std == (0.4, 0.2)
