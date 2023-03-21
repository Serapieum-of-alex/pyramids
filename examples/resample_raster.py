"""resample raster from 4000 m cell size to 100 m """
from pyramids.raster import Raster

src_path = "examples/data/acc4000.tif"
#%%
src = Raster.read(src_path)
arr = src.ReadAsArray()
print(f"current cell size is {src.cell_size} m")

resample_raster_cell_size = 100
resample_raster_resample_technique = "bilinear"

dst = src.resample(
    resample_raster_cell_size,
    method=resample_raster_resample_technique,
)
