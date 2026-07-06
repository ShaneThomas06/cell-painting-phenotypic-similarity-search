from cell_painting_profiling.data.s3_listing import (
    build_public_s3_list_url,
    parse_s3_listing,
)


def test_build_public_s3_list_url_encodes_prefix():
    url = build_public_s3_list_url(
        bucket="cellpainting-gallery",
        prefix="cpg0002-jump-scope/source_4/",
    )

    assert url.startswith("https://cellpainting-gallery.s3.amazonaws.com/?")
    assert "list-type=2" in url
    assert "prefix=cpg0002-jump-scope%2Fsource_4%2F" in url
    assert "delimiter=%2F" in url


def test_parse_s3_listing_reads_prefixes_and_objects():
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <CommonPrefixes><Prefix>dataset/images/</Prefix></CommonPrefixes>
      <Contents><Key>dataset/metadata.csv</Key><Size>123</Size></Contents>
    </ListBucketResult>
    """

    listing = parse_s3_listing(xml)

    assert listing.prefixes == ["dataset/images/"]
    assert listing.objects[0].key == "dataset/metadata.csv"
    assert listing.objects[0].size == 123
