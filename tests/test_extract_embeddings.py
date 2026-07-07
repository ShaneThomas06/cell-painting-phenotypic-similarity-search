import torch

from cell_painting_profiling.embeddings.extract import ResNetEmbeddingModel


def test_resnet_embedding_model_returns_512_features():
    model = ResNetEmbeddingModel(num_input_channels=5)
    model.eval()
    images = torch.zeros((2, 5, 64, 64))

    with torch.no_grad():
        embeddings = model(images)

    assert embeddings.shape == (2, 512)
