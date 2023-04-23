# import numpy as np
from pyramids.dataset import Dataset

Path = f"examples/data/DEM5km_Rhine_burned_acc.tif"
# data = np.loadtxt(Path, delimiter=',')
dataset = Dataset.read_file(Path)
# %%
lowervalue = 0.1  # DataArr[DataArr != NoDataValue].min()
uppervalue = 20  # DataArr[DataArr != NoDataValue].max()

ClusterArray, count, Position, Values = dataset.cluster(lowervalue, uppervalue)
