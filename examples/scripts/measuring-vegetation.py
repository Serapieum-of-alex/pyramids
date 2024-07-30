from pyramids.dataset import Dataset
from osgeo_utils import gdal_calc
import pandas as pd

# path = r"\\MYCLOUDEX2ULTRA\satellite-data\landsat\lake-taho"
path = r"examples\data\landsat\lake-taho"
# %%
color_palette = pd.read_csv(f"{path}/beige_green.txt")
# %%
b4 = Dataset.read_file(rf"{path}\LC08_L2SP_043033_20210922_20210930_02_T1_SR_B4.TIF")
print(b4)
# b4.plot(color_scale="linear")
b4.no_data_value
b5 = Dataset.read_file(rf"{path}\LC08_L2SP_043033_20210922_20210930_02_T1_SR_B5.TIF")
# b5.plot(color_scale="linear")
b5.no_data_value
# %%
"""
gdal_calc.py
-D LC08_L2SP_043033_20210922_20210930_02_T1_SR_B4.TIF
-E LC08_L2SP_043033_20210922_20210930_02_T1_SR_B5.TIF
--calc="((E * 0.0000275 - 0.2) - (D * 0.0000275 - 0.2))/((E * 0.0000275 - 0.2) + (D * 0.0000275 - 0.2))"
--outfile=tahoe_LC08_20210922_SR_NDVI.tif
--type=Float32
--co COMPRESS=DEFLATE
--co PREDICTOR=2
"""
# %%
"--calc and enclosed in double quotes. The operations are performed on the bands defined previously, and can be any function available in NumPy"
# gdal_calc.py will write the output file in the same format as the input files.
ndvi = gdal_calc.Calc(
    calc="((E * 0.0000275 - 0.2) - (D * 0.0000275 - 0.2))/((E * 0.0000275 - 0.2) + (D * 0.0000275 - 0.2))",
    # calc="A+B",
    D=b4.raster,
    E=b5.raster,
    format="MEM",
    outfile=r"tahoe_LC08_20210922_SR_NDVI.tif",
    type="Float32",
    creation_options=["COMPRESS=DEFLATE", "PREDICTOR=2"],
)
# %%
ndvi = Dataset(ndvi)
print(ndvi)
# color_scale="boundary-norm", bounds=[0, 0.2, 0.4, 0.6, 0.8, 1]
ndvi.plot(vmin=0, vmax=1, color_scale="linear")
ndvi.stats()
ndvi = ndvi.change_no_data_value(-9999, ndvi.no_data_value[0])

ndvi.to_file(rf"{path}\tahoe_LC08_20210922_SR_NDVI.tif")
color_relief = ndvi.color_relief(band=0, color_table=color_palette)
color_relief.to_file(f"{path}/tahoe_LC08_20210922_SR_NDVI_color_relief.tif")
# %%
# ndvi = Dataset.read_file(r"examples\data\landsat\lake-taho\tahoe_LC08_20210922_SR_NDVI.tif", read_only=False)

color_relief.no_data_value = 0
color_relief.plot(rgb=[0, 1, 2, 3])
color_relief.read_array(band=0, window=[0, 0, 5, 5])
