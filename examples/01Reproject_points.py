"""
Created on Fri Feb 19 17:57:06 2021

@author: mofarrag
"""
from Hapi.gis.vector import Vector

#%%
lon = [-180, -179.5]
lat = [90, 90]

from_epsg = 4326
to_epsg = 32618

y, x = Vector.ReprojectPoints(lat, lon, from_epsg, to_epsg, precision=9)

#%% brazil
x = 4522693.11
y = 7423522.55
from_epsg = 5641
to_epsg = 4326

lat, lon = Vector.ReprojectPoints([y], [x], from_epsg, to_epsg, precision=4)

assert lat[0] == -22.6895 and lon[0] == -47.2903, "Error ReprojectPoints error 1"
#%%

lon = [-32, 71]
lat = [32.0, 83]
from_epsg = 4326
to_epsg = 4647

y, x = Vector.ReprojectPoints(lat, lon, from_epsg, to_epsg, precision=4)

assert y[0] == 4390682.5383 and y[1] == 9629641.4604, "Error ReprojectPoints error 2y"
assert x[0] == 28494364.9445 and x[1] == 33190988.6123, "Error ReprojectPoints error 2x"
