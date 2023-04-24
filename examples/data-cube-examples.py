import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from pyramids.dataset import Datacube

path = "tests/data/raster-folder-coello"
#%%
cube = Datacube.read_separate_files(path)
cube.read_dataset()
# arr = cube.data
# np.save(f"{path}/coello.npy",arr)
cube.plot()
