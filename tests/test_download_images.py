from cell_painting_profiling.data.download_images import s3_url_to_https


def test_s3_url_to_https_converts_cellpainting_gallery_url():
    url = "s3://cellpainting-gallery/cpg0002-jump-scope/file.tif"

    converted = s3_url_to_https(url)

    assert converted == "https://cellpainting-gallery.s3.amazonaws.com/cpg0002-jump-scope/file.tif"
