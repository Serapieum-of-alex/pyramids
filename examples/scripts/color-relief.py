from pyramids.dataset import Dataset
import pandas as pd

# %%
path = r"\\MYCLOUDEX2ULTRA\satellite-data\DEM"
color_file = f"daymet_v4_temperature_palette_-20_40.txt"
output_path = f"daymet_v4_tmax_annavg_na_2022_color99999.tif"
# %%
max_dataset = Dataset.read_file(f"{path}/daymet_v4_tmax_annavg_na_2022.tif")
# max_dataset.plot(cmap="RdYlBu")
df = pd.read_csv(color_file, header=None)
df.columns = ["values", "red", "green", "blue", "alpha"]
color_relief = max_dataset.color_relief(band=0, color_table=df)
color_relief.plot(rgb=[0, 1, 2, 3])
# %%
dataset = Dataset.read_file(f"{path}/CHELSA_tas_09_1981-2010_V.2.1.tif")
dataset.create_overviews()
dataset.plot(overview_index=0)
# %%
min_dataset = Dataset.read_file(f"{path}/daymet_v4_tmin_annavg_na_2022.tif")
min_dataset.plot(cmap="RdYlBu")
