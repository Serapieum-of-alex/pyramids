from pyramids.dataset import Dataset

# %%
import numpy as np

rng = np.random.default_rng(0)
arr = rng.integers(1, 10, size=(2, 5, 5))
top_left_corner = (0, 0)
cell_size = 0.05
dataset = Dataset.create_from_array(
    arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
)

df = dataset.to_xyz(bands=[0, 1])
print(df)
