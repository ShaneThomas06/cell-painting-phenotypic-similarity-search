from torchvision import transforms


def build_image_transforms(image_size: int = 224, train: bool = True):
    """Build default image transforms for baseline experiments."""
    transform_steps = []
    if train:
        transform_steps.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        transform_steps.extend(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
        ]
    )
    return transforms.Compose(transform_steps)
