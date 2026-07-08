import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cell_painting_profiling.training.train import (
    assign_compound_holdout_split,
    evaluate_classifier,
    train_one_epoch,
)


class TinyImageDataset(Dataset):
    def __init__(self):
        self.images = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.8, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 1.0]]],
                [[[0.0, 0.0], [0.0, 0.9]]],
            ]
        )
        self.labels = torch.tensor([0, 0, 1, 1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"image": self.images[index], "label": self.labels[index]}


def tiny_collate(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }


def test_assign_compound_holdout_split_holds_out_one_compound_per_mechanism():
    manifest = pd.DataFrame(
        {
            "image_record_id": [f"id_{i}" for i in range(8)],
            "perturbation_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "compound_name": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "mechanism_of_action": ["m1", "m1", "m1", "m1", "m2", "m2", "m2", "m2"],
        }
    )

    split = assign_compound_holdout_split(manifest)

    val = split.loc[split["split"] == "val"]
    train = split.loc[split["split"] == "train"]
    assert set(val["perturbation_id"]) == {"b", "d"}
    assert set(train["perturbation_id"]) == {"a", "c"}


def test_train_one_epoch_updates_tiny_model():
    loader = DataLoader(TinyImageDataset(), batch_size=2, collate_fn=tiny_collate)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    before = model[1].weight.detach().clone()

    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        torch.device("cpu"),
    )

    assert metrics["loss"] > 0
    assert not torch.equal(before, model[1].weight.detach())


def test_evaluate_classifier_returns_metrics():
    loader = DataLoader(TinyImageDataset(), batch_size=2, collate_fn=tiny_collate)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate_classifier(
        model,
        loader,
        criterion,
        torch.device("cpu"),
    )

    assert set(metrics) == {"loss", "accuracy", "balanced_accuracy", "macro_f1"}
