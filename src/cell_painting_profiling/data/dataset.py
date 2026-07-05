from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_METADATA_COLUMNS = {"image_path", "perturbation", "batch"}


@dataclass(frozen=True)
class RxRx1Record:
    image_path: Path
    perturbation: str
    batch: str
    cell_type: str | None = None


class CellPaintingDataset(Dataset):
    """Dataset wrapper for Cell Painting image records."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        image_root: str | Path,
        transform: Callable | None = None,
    ) -> None:
        missing = REQUIRED_METADATA_COLUMNS.difference(metadata.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Metadata is missing required columns: {missing_text}")

        self.metadata = metadata.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.label_to_index = {
            label: index
            for index, label in enumerate(sorted(self.metadata["perturbation"].unique()))
        }

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict:
        row = self.metadata.iloc[index]
        image_path = self.image_root / row["image_path"]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        label = self.label_to_index[row["perturbation"]]
        return {
            "image": image,
            "label": label,
            "perturbation": row["perturbation"],
            "batch": row["batch"],
            "image_path": str(image_path),
        }


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load RxRx1-style metadata from CSV or Parquet."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
