# Setup

Use Python 3.10, 3.11, 3.12, or 3.13 for this project. The deep-learning stack depends on PyTorch and torchvision, which may lag behind the newest Python releases.

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Verify

```bash
pytest
```

## Metadata Inspection

After downloading or preparing an RxRx1-style metadata table:

```bash
inspect-cell-painting-metadata --metadata data/raw/metadata.csv --output reports/tables/metadata_summary.csv
```

Raw RxRx1 image data is large and should stay outside Git history. Keep downloaded data under `data/raw/`, which is ignored except for placeholder files.
