from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd


def s3_url_to_https(url: str) -> str:
    """Convert an s3://cellpainting-gallery URL to an anonymous HTTPS URL."""
    prefix = "s3://cellpainting-gallery/"
    if not url.startswith(prefix):
        return url
    key = url.removeprefix(prefix)
    return f"https://cellpainting-gallery.s3.amazonaws.com/{key}"


def download_manifest_images(manifest: pd.DataFrame, overwrite: bool = False) -> int:
    """Download all images described by a channel-level manifest."""
    required = {"source_url", "local_path"}
    missing = required.difference(manifest.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Manifest is missing required columns: {missing_text}")

    downloaded = 0
    for _, row in manifest.iterrows():
        destination = Path(row["local_path"])
        if destination.exists() and not overwrite:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_url = s3_url_to_https(row["source_url"])
        urllib.request.urlretrieve(source_url, destination)
        downloaded += 1
    return downloaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download images from a manifest CSV.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = pd.read_csv(args.manifest)
    downloaded = download_manifest_images(manifest, overwrite=args.overwrite)
    print(f"Downloaded {downloaded} images from {args.manifest}")


if __name__ == "__main__":
    main()
