import os

from pyramids.utils import extractFromGZ


def test_extractFromGZ(
    compressed_raster: str,
    uncompressed_output: str,
):
    extractFromGZ(compressed_raster, uncompressed_output, delete=False)
    assert os.path.exists(uncompressed_output)
    # deleting the uncompressed file
    os.remove(uncompressed_output)
