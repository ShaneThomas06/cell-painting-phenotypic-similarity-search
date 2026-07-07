import numpy as np
import pandas as pd
import torch
from PIL import Image

from cell_painting_profiling.data.multichannel_dataset import (
    DEFAULT_CHANNEL_ORDER,
    MultiChannelCellPaintingDataset,
)


def test_multichannel_dataset_loads_stacked_tensor(tmp_path):
    rows = []
    for index, channel in enumerate(DEFAULT_CHANNEL_ORDER, start=1):
        path = tmp_path / f"{channel}.tif"
        Image.fromarray(np.full((8, 8), index, dtype=np.uint16)).save(path)
        rows.append(
            {
                "image_record_id": "record_1",
                "channel": channel,
                "local_path": str(path),
                "perturbation_id": "BRD-A",
                "compound_name": "compound_a",
                "mechanism_of_action": "HDAC inhibitor",
                "well": "A01",
                "site": 1,
            }
        )

    dataset = MultiChannelCellPaintingDataset(pd.DataFrame(rows), image_size=4)
    item = dataset[0]

    assert len(dataset) == 1
    assert item["image"].shape == (5, 4, 4)
    assert item["mechanism_of_action"] == "HDAC inhibitor"
    assert torch.all(item["image"] >= 0)
    assert torch.all(item["image"] <= 1)


def test_multichannel_dataset_requires_all_channels(tmp_path):
    path = tmp_path / "rna.tif"
    Image.fromarray(np.ones((8, 8), dtype=np.uint16)).save(path)
    manifest = pd.DataFrame(
        {
            "image_record_id": ["record_1"],
            "channel": ["rna"],
            "local_path": [str(path)],
            "perturbation_id": ["BRD-A"],
            "compound_name": ["compound_a"],
            "mechanism_of_action": ["HDAC inhibitor"],
        }
    )

    try:
        MultiChannelCellPaintingDataset(manifest)
    except ValueError as exc:
        assert "missing channels" in str(exc)
    else:
        raise AssertionError("Expected missing-channel validation to fail")
