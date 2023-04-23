import os
from typing import List

import pytest
from osgeo import gdal
from pyramids._errors import FileFormatNoSupported
from pyramids._io import _parse_path, extract_from_gz, read_file


class TestParsePath:
    def test_one_compressed_zip_file(
        self,
        one_compressed_file_zip: str,
        multiple_compressed_file_zip_content: List[str],
    ):
        first_file = multiple_compressed_file_zip_content[0]
        """zip file contzins only one compressed file"""
        res = _parse_path(one_compressed_file_zip)
        assert res == f"/vsizip/{one_compressed_file_zip}/{first_file}"

    def test_multiple_compressed_zip_file(
        self,
        multiple_compressed_file_zip: str,
        multiple_compressed_file_zip_content: List[str],
    ):
        first_file = multiple_compressed_file_zip_content[0]
        second_file = multiple_compressed_file_zip_content[1]
        """zip file contzins multiple compressed file"""
        res = _parse_path(multiple_compressed_file_zip, file_i=0)
        assert res == f"/vsizip/{multiple_compressed_file_zip}/{first_file}"
        res = _parse_path(multiple_compressed_file_zip, file_i=1)
        assert res == f"/vsizip/{multiple_compressed_file_zip}/{second_file}"

    def test_give_path_inside_zip_file(
        self,
        multiple_compressed_file_zip: str,
        multiple_compressed_file_zip_content: List[str],
    ):
        first_file = multiple_compressed_file_zip_content[0]
        """zip file contzins multiple compressed file"""
        res = _parse_path(f"{multiple_compressed_file_zip}/{first_file}")
        assert res == f"/vsizip/{multiple_compressed_file_zip}/{first_file}"

    def test_one_compressed_gzip_file(self, one_compressed_file_gzip: str):
        """zip file contzins only one compressed file"""
        res = _parse_path(one_compressed_file_gzip)
        assert res == f"/vsigzip/{one_compressed_file_gzip}"

    def test_multiple_compressed_gzip_file(self, multiple_compressed_file_gzip: str):
        """zip file contzins only one compressed file"""
        res = _parse_path(multiple_compressed_file_gzip)
        assert res == f"/vsigzip/{multiple_compressed_file_gzip}"


def test_extract_from_gz(
    one_compressed_file_gzip,
    unzip_gzip_file_name,
):
    extract_from_gz(one_compressed_file_gzip, unzip_gzip_file_name, delete=False)
    assert os.path.exists(unzip_gzip_file_name)
    # deleting the uncompressed file
    os.remove(unzip_gzip_file_name)


class TestReadZip:
    def test_read_single_compressed_zip(self, one_compressed_file_zip: str):
        src = read_file(one_compressed_file_zip)
        assert isinstance(src, gdal.Dataset)

    def test_multiple_compressed_zip_file(self, multiple_compressed_file_zip: str):
        src = read_file(multiple_compressed_file_zip)
        assert isinstance(src, gdal.Dataset)


class TestReadGzip:
    def test_read_single_compressed_gzip(self, one_compressed_file_gzip: str):
        src = read_file(one_compressed_file_gzip)
        assert isinstance(src, gdal.Dataset)

    def test_multiple_compressed_gzip_file_error(
        self, multiple_compressed_file_gzip: str
    ):
        try:
            read_file(multiple_compressed_file_gzip)
        except FileFormatNoSupported:
            pass

    def test_multiple_compressed_gzip_file_with_internal_path(
        self,
        multiple_compressed_file_gzip: str,
        multiple_compressed_file_gzip_content: List[str],
    ):
        first_file = multiple_compressed_file_gzip_content[0]
        try:
            read_file(f"{multiple_compressed_file_gzip}/{first_file}")
        except FileFormatNoSupported:
            pass


class TestReadTar:
    def test_read_single_compressed(self, one_compressed_file_tar: str):
        src = read_file(one_compressed_file_tar)
        assert isinstance(src, gdal.Dataset)

    def test_multiple_compressed_tar_file_with_internal_path(
        self,
        multiple_compressed_file_tar: str,
        multiple_compressed_file_gzip_content: List[str],
    ):
        first_file = multiple_compressed_file_gzip_content[0]
        src = read_file(f"{multiple_compressed_file_tar}/{first_file}")
        assert isinstance(src, gdal.Dataset)


class TestHttpsrequest:
    @pytest.mark.vfs
    def test_read_from_aws(self):
        url = (
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/31/U/FU/2020/3"
            "/S2A_31UFU_20200328_0_L2A/B01.tif"
        )
        src = read_file(url)
        assert src.RasterXSize == 1830
        assert src.RasterYSize == 1830
        assert isinstance(src, gdal.Dataset)
