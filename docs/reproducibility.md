# Clean-clone verification

Use Python 3.12 (the CI version). From a terminal:

```sh
git clone https://github.com/ShaneThomas06/cell-painting-phenotypic-similarity-search.git
cd cell-painting-phenotypic-similarity-search
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then run:

```sh
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
python -m ruff check src tests
python -m pytest -q
inspect-cell-painting-metadata --help
```

Expected: lint and tests pass, and the final command prints usage and exits
successfully. These checks require no study downloads, GPU, or model weights;
installation requires internet access. Tests use small synthetic fixtures and
temporary outputs. They do not regenerate or overwrite tracked scientific results.
CI runs the same checks on every push and pull request.

This is a software smoke check, not a reproduction of the complete study.
For an exact environment record, save `python -m pip freeze > environment.txt`
alongside your own run outputs, and record the repository commit and input checksums.
Version bounds describe supported dependency families, not an exact lockfile.

## Full benchmark

See [setup](setup.md) and the README for processed manifests, raw image downloads,
training and figure-generation steps. Pretrained experiments may download weights.
The synthetic tests use randomly initialized models. PyTorch and torchvision must
be installed as a compatible pair; pip resolves torchvision's torch requirement.
The bounded dependencies and command entry points are defined only in
`pyproject.toml`; `requirements.txt` delegates to the same development install.
