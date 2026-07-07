import torch.nn as nn
from torchvision import models


def adapt_first_conv(model: nn.Module, num_input_channels: int) -> nn.Module:
    """Adapt a torchvision ResNet first convolution to non-RGB input channels."""
    conv = model.conv1
    if conv.in_channels == num_input_channels:
        return model

    new_conv = nn.Conv2d(
        num_input_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    model.conv1 = new_conv
    return model


def build_resnet18_classifier(
    num_classes: int,
    num_input_channels: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    """Build a ResNet18 classifier with a configurable input channel count."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model = adapt_first_conv(model, num_input_channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
