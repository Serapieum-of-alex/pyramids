from pyramids.dataset import Dataset

path = r"\\MYCLOUDEX2ULTRA\satellite-data\DEM"
# dem_path = "USGS_1M_10_x56y512_WA_FEMAHQ_2018_D18.tif"
file_id = "1q6wRKTtTe0NVanawduPeOE3WUhgsyAoM"
# dem_path = f"/vsicurl/https://drive.google.com/uc?export=download&id={FILE_ID}"
dem_path = f"/vsicurl/https://drive.google.com/uc?export=download&id={file_id}"
# %%
dataset = Dataset.read_file(f"{dem_path}")
print(dataset)
dataset.plot(color_scale="linear")
dataset.stats()
src = dataset.raster
# %%
# Define input and output file paths
hill_shade = dataset.hill_shade(
    band=0,
    light_source_elevation=45,
    light_source_angle=315,
    vertical_exaggeration=1,
    scale=1,
)

hill_shade.plot(cmap="gray")
# %%
hill_shade = dataset.hill_shade(
    band=0,
    light_source_elevation=45,
    light_source_angle=None,
    vertical_exaggeration=1,
    scale=1,
    multiDirectional=True,
)

hill_shade.plot(cmap="gray")
# hill_shade.to_file(f"{path}/ssss.tif")
# hill_shade.close()
