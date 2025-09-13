from pyramids.netcdf.dimensions import (
    _strip_braces, _smart_split_csv, _coerce_scalar, _parse_values_list, DimensionsIndex, Dimension,
    parse_gdal_netcdf_dimensions
)


class TestStripBraces:
    def test_with_braces(self):
        """Input has surrounding braces -> returns content without braces."""
        assert _strip_braces("{1,2,3}") == "1,2,3"

    def test_without_braces(self):
        """Input has no braces -> returns original string stripped."""
        assert _strip_braces(" 1,2,3 ") == "1,2,3"

    def test_empty_braces(self):
        """Input is empty braces -> returns empty string."""
        assert _strip_braces("{}") == ""


class TestSmartSplitCsv:
    def test_simple_list(self):
        """Comma-separated values with braces -> split into list of strings."""
        assert _smart_split_csv("{a,b,c}") == ["a", "b", "c"]

    def test_with_spaces(self):
        """Values with spaces around commas -> trimmed in result."""
        assert _smart_split_csv("{ a , b , c }") == ["a", "b", "c"]

    def test_empty_content(self):
        """Empty braces -> returns empty list."""
        assert _smart_split_csv("{}") == []

    def test_no_braces(self):
        """No braces -> still split by comma."""
        assert _smart_split_csv("a, b") == ["a", "b"]


class TestCoerceScalar:
    def test_integer(self):
        """String integer -> coerced to int."""
        assert _coerce_scalar("42") == 42

    def test_negative_integer(self):
        """Negative integer string -> coerced to int."""
        assert _coerce_scalar("-5") == -5

    def test_float(self):
        """Float string -> coerced to float."""
        assert _coerce_scalar("3.14") == 3.14

    def test_non_number(self):
        """Non-numeric string -> returned unchanged."""
        assert _coerce_scalar("abc") == "abc"


class TestParseValuesList:
    def test_mixed_values(self):
        """Braced CSV with ints and floats -> coerced correctly."""
        result = _parse_values_list("{1, 2.5, abc}")
        assert result == [1, 2.5, "abc"]


class TestNetCDFDimensionsIndexFromMetadata:
    def test_basic_dimensions(self):
        """Metadata with DEF and VALUES -> parse into dimensions with size and values."""
        md = {
            'NETCDF_DIM_EXTRA': '{time,level0}',
            'NETCDF_DIM_level0_DEF': '{3,6}',
            'NETCDF_DIM_level0_VALUES': '{1,2,3}',
            'NETCDF_DIM_time_DEF': '{2,6}',
            'NETCDF_DIM_time_VALUES': '{0,31}',
        }
        idx = DimensionsIndex.from_metadata(md)
        assert set(idx.names) == {"time", "level0"}
        assert idx["time"].size == 2
        assert idx["level0"].values == [1, 2, 3]

    def test_missing_def_infer_size_from_values(self):
        """If DEF missing, size inferred from VALUES length."""
        md = {
            'NETCDF_DIM_var_VALUES': '{10,20,30}'
        }
        idx = DimensionsIndex.from_metadata(md)
        assert idx["var"].size == 3

    def test_missing_values_size_from_def(self):
        """If VALUES missing, size comes from DEF first value."""
        md = {
            'NETCDF_DIM_var_DEF': '{5,100}'
        }
        idx = DimensionsIndex.from_metadata(md)
        assert idx["var"].size == 5

    def test_non_prefixed_keys_ignored(self):
        """Keys without NETCDF_DIM_ prefix are ignored."""
        md = {
            'OTHER_KEY': 'abc'
        }
        idx = DimensionsIndex.from_metadata(md)
        assert idx.names == []

    def test_extra_key_only(self):
        """EXTRA entry with list of dimensions -> dimensions created."""
        md = {
            'NETCDF_DIM_EXTRA': '{dim1,dim2}'
        }
        idx = DimensionsIndex.from_metadata(md)
        assert set(idx.names) == {"dim1", "dim2"}

    def test_def_with_non_ints(self):
        """DEF entry with non-integer tokens -> ignored in def_fields."""
        md = {
            'NETCDF_DIM_var_DEF': '{a,2}'
        }
        idx = DimensionsIndex.from_metadata(md)
        assert idx["var"].def_fields == (2,)
        assert idx["var"].size == 2


class TestNetCDFDimensionsIndexMethods:
    def test_len_and_iter(self):
        """Index supports len() and iteration over Dimension."""
        md = {
            'NETCDF_DIM_var_VALUES': '{1,2}'
        }
        idx = DimensionsIndex.from_metadata(md)
        assert len(idx) == 1
        dims = list(iter(idx))
        assert isinstance(dims[0], Dimension)

    def test_contains_and_getitem(self):
        """Supports membership test and item access by name."""
        md = {
            'NETCDF_DIM_var_VALUES': '{1}'
        }
        idx = DimensionsIndex.from_metadata(md)
        assert "var" in idx
        assert idx["var"].name == "var"

    def test_to_dict(self):
        """Serializes dimensions to dictionary structure."""
        md = {
            'NETCDF_DIM_var_VALUES': '{1,2,3}'
        }
        idx = DimensionsIndex.from_metadata(md)
        d = idx.to_dict()
        assert d["var"]["size"] == 3
        assert d["var"]["values"] == [1, 2, 3]


class TestParseGdalNetcdfDimensions:
    def test_wrapper_function(self):
        """Convenience wrapper delegates to from_metadata."""
        md = {
            'NETCDF_DIM_var_VALUES': '{4,5}'
        }
        idx = parse_gdal_netcdf_dimensions(md)
        assert idx["var"].values == [4, 5]
