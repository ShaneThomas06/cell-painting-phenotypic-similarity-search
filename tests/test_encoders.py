from cell_painting_profiling.models.encoders import build_resnet18_classifier


def test_build_resnet18_classifier_accepts_five_channels():
    model = build_resnet18_classifier(
        num_classes=4,
        num_input_channels=5,
        pretrained=False,
    )

    assert model.conv1.in_channels == 5
    assert model.fc.out_features == 4
