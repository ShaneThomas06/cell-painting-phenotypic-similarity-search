from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def validate_manifest_images(manifest: pd.DataFrame) -> pd.DataFrame:
    """Validate that downloaded images exist and can be opened by Pillow."""
    rows = []
    for _, row in manifest.iterrows():
        path = Path(row["local_path"])
        status = {
            "local_path": str(path),
            "exists": path.exists(),
            "width": None,
            "height": None,
            "mode": None,
            "error": None,
        }
        if path.exists():
            try:
                with Image.open(path) as image:
                    status["width"], status["height"] = image.size
                    status["mode"] = image.mode
            except Exception as exc:  # pragma: no cover - defensive reporting
                status["error"] = str(exc)
        rows.append(status)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate downloaded image files.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = pd.read_csv(args.manifest)
    report = validate_manifest_images(manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    failures = report.loc[(~report["exists"]) | report["error"].notna()]
    print(f"Validated {len(report)} images; failures: {len(failures)}")


if __name__ == "__main__":
    main()
