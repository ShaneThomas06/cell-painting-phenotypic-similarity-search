import torch

from cell_painting_profiling.embeddings.extract import (
    ResNetEmbeddingModel,
    load_trained_embedding_model,
)
from cell_painting_profiling.models.encoders import build_resnet18_classifier


def test_resnet_embedding_model_returns_512_features():
    model = ResNetEmbeddingModel(num_input_channels=5)
    model.eval()
    images = torch.zeros((2, 5, 64, 64))

    with torch.no_grad():
        embeddings = model(images)

    assert embeddings.shape == (2, 512)


def test_load_trained_embedding_model_returns_512_features(tmp_path):
    classifier = build_resnet18_classifier(
        num_classes=2,
        num_input_channels=5,
        pretrained=False,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": classifier.state_dict(),
            "label_map": {"m1": 0, "m2": 1},
            "channel_order": ["rna", "mito", "agp", "er", "dna"],
            "image_size": 64,
        },
        checkpoint_path,
    )

    model, checkpoint = load_trained_embedding_model(checkpoint_path, torch.device("cpu"))
    images = torch.zeros((2, 5, 64, 64))

    with torch.no_grad():
        embeddings = model(images)

    assert checkpoint["image_size"] == 64
    assert embeddings.shape == (2, 512)

def test_resnet_embedding_model_accepts_pretrained_flag(monkeypatch):
    calls = {}

    def fake_builder(num_classes, num_input_channels, pretrained):
        calls["pretrained"] = pretrained
        return build_resnet18_classifier(
            num_classes=num_classes,
            num_input_channels=num_input_channels,
            pretrained=False,
        )

    monkeypatch.setattr(
        "cell_painting_profiling.embeddings.extract.build_resnet18_classifier",
        fake_builder,
    )

    model = ResNetEmbeddingModel(num_input_channels=5, pretrained=True)

    assert calls["pretrained"] is True
    assert model(torch.zeros((1, 5, 64, 64))).shape == (1, 512)

