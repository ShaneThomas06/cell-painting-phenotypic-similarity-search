import numpy as np
import pandas as pd
from PIL import Image

from cell_painting_profiling.data.channel_stats import compute_channel_stats
from cell_painting_profiling.data.multichannel_dataset import DEFAULT_CHANNEL_ORDER


def test_compute_channel_stats_reports_mean_std_by_channel(tmp_path):
    rows = []
    values = {channel: index for index, channel in enumerate(DEFAULT_CHANNEL_ORDER, start=1)}
    for channel, value in values.items():
        path = tmp_path / f"{channel}.tif"
        Image.fromarray(np.full((4, 4), value, dtype=np.uint16)).save(path)
        rows.append({"channel": channel, "local_path": str(path)})

    stats = compute_channel_stats(pd.DataFrame(rows))

    assert stats["channel_order"] == list(DEFAULT_CHANNEL_ORDER)
    for channel, value in values.items():
        expected = value / np.iinfo(np.uint16).max
        assert stats["channels"][channel]["num_files"] == 1
        assert np.isclose(stats["channels"][channel]["mean"], expected)
        assert stats["channels"][channel]["std"] == 0.0

