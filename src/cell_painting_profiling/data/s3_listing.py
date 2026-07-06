from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass


S3_XML_NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int


@dataclass(frozen=True)
class S3Listing:
    prefixes: list[str]
    objects: list[S3Object]


def build_public_s3_list_url(
    bucket: str,
    prefix: str,
    delimiter: str | None = "/",
) -> str:
    """Build a public S3 ListObjectsV2 URL."""
    params = {"list-type": "2", "prefix": prefix}
    if delimiter is not None:
        params["delimiter"] = delimiter
    return f"https://{bucket}.s3.amazonaws.com/?{urllib.parse.urlencode(params)}"


def parse_s3_listing(xml_bytes: bytes) -> S3Listing:
    """Parse the XML response from S3 ListObjectsV2."""
    root = ET.fromstring(xml_bytes)
    prefixes = [
        node.find(f"{S3_XML_NAMESPACE}Prefix").text
        for node in root.findall(f"{S3_XML_NAMESPACE}CommonPrefixes")
    ]
    objects = [
        S3Object(
            key=node.find(f"{S3_XML_NAMESPACE}Key").text,
            size=int(node.find(f"{S3_XML_NAMESPACE}Size").text),
        )
        for node in root.findall(f"{S3_XML_NAMESPACE}Contents")
    ]
    return S3Listing(prefixes=prefixes, objects=objects)


def list_public_s3_prefix(
    bucket: str,
    prefix: str,
    delimiter: str | None = "/",
    timeout: int = 30,
) -> S3Listing:
    """List prefixes and objects from a public S3 bucket without AWS credentials."""
    url = build_public_s3_list_url(bucket=bucket, prefix=prefix, delimiter=delimiter)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return parse_s3_listing(response.read())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List a public S3 prefix.")
    parser.add_argument("--bucket", default="cellpainting-gallery")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--no-delimiter", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    delimiter = None if args.no_delimiter else "/"
    listing = list_public_s3_prefix(
        bucket=args.bucket,
        prefix=args.prefix,
        delimiter=delimiter,
    )
    for prefix in listing.prefixes:
        print(f"DIR\t{prefix}")
    for obj in listing.objects:
        print(f"FILE\t{obj.size}\t{obj.key}")


if __name__ == "__main__":
    main()
