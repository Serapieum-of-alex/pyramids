import matplotlib

matplotlib.use("TkAgg")
from pyramids.dataset import Dataset

geotiff_path = "examples/data/dem/DEM5km_Rhine_burned_fill.tif"
ascii_path = "examples/data/dem/dem5km_rhine.asc"
nc_path = "examples/data/dem/dem5km_rhine.nc"
#%%
dataset = Dataset.read_file(geotiff_path)
dataset.plot(
    title="Rhine river basin",
    ticks_spacing=500,
    cmap="terrain",
    color_scale=1,
    vmin=0,
    cbar_label="Elevation (m)",
)

dataset.raster
dataset.cell_size
dataset.values
dataset.shape
dataset.rows
dataset.columns
dataset.pivot_point
dataset.geotransform
dataset.epsg
dataset.bounds
dataset.bounds.plot()
dataset.bbox
dataset.crs
f = dataset.lat
dataset.lon
dataset.x
f = dataset.y

dataset.band_count
dataset.band_names
dataset.variables
dataset.no_data_value
dataset.meta_data
dataset.dtype
dataset.file_name
dataset.time_stamp
dataset.driver_type

# dataset.to_file("examples/data/dem/dem5km_rhine.asc", driver="ascii")
# dataset.to_file("examples/data/dem/dem5km_rhine.nc", driver="netcdf")
#%%
dataset = Dataset.read_file(ascii_path)
dataset.plot(
    title="Rhine river basin",
    ticks_spacing=500,
    cmap="terrain",
    color_scale=1,
    vmin=0,
    cbar_label="Elevation (m)",
)
#%%
dataset = Dataset.read_file(nc_path)
dataset.plot(
    title="Rhine river basin",
    ticks_spacing=500,
    cmap="terrain",
    color_scale=1,
    vmin=0,
    cbar_label="Elevation (m)",
)
