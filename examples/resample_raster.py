"""resample raster from 4000 m cell size to 100 m """
from pyramids.raster import Raster

src_path = "examples/data/acc4000.tif"
#%%
src = Raster.openDataset(src_path)
arr = src.ReadAsArray()
ros, cos, proj, bands, gt, no_data_value, dtype = Raster.getRasterDetails(src)
print(f"current cell size is {gt[1]} m")

resample_raster_cell_size = 100
resample_raster_resample_technique = "bilinear"

dst = Raster.resampleRaster(
    src,
    resample_raster_cell_size,
    resample_technique=resample_raster_resample_technique,
)
