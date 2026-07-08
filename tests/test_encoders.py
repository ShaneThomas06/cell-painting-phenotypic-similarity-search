import torch
import torch.nn as nn

from cell_painting_profiling.models.encoders import (
    adapt_first_conv,
    build_resnet18_classifier,
)


def test_build_resnet18_classifier_accepts_five_channels():
    model = build_resnet18_classifier(
        num_classes=4,
        num_input_channels=5,
        pretrained=False,
    )

    assert model.conv1.in_channels == 5
    assert model.fc.out_features == 4


def test_adapt_first_conv_spreads_rgb_weights_to_extra_channels():
    model = nn.Module()
    model.conv1 = nn.Conv2d(3, 2, kernel_size=1, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(torch.tensor([[[[1.0]], [[2.0]], [[3.0]]], [[[4.0]], [[5.0]], [[6.0]]]]))

    adapted = adapt_first_conv(model, num_input_channels=5)

    assert adapted.conv1.in_channels == 5
    assert torch.allclose(
        adapted.conv1.weight[0, :, 0, 0],
        torch.full((5,), 1.2),
    )
    assert torch.allclose(
        adapted.conv1.weight[1, :, 0, 0],
        torch.full((5,), 3.0),
    )
