import torch

from cell_painting_profiling.data.transforms import (
    ChannelStackTransform,
    augment_channel_stack,
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
