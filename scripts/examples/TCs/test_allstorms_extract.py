#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import sys

# pip
import numpy as np
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0, op.join(op.dirname(__file__), '..', '..', '..'))

# teslakit
from teslakit.database import Database
from teslakit.storms import Extract_Circle



# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')


# --------------------------------------
# load data and set parameters
xds_wmo = db.Load_TCs_noaa()  # noaa Allstorms.ibtracs_wmo

# wave point longitude and latitude
pnt_lon = 167.5
pnt_lat = 9.75

# radius for TCs selection (º)
r1 = 14

# TODO: CON RADIO 14, la 251 o 253 tiene valores mal de presion

# --------------------------------------
# Select Historical TCs inside circle

# dictionary with needed variable names
d_vns = {
    'longitude': 'lon_wmo',
    'latitude': 'lat_wmo',
    'time': 'time_wmo',
    'pressure': 'pres_wmo',
}

# Select TCs that crosses a circular area R1
xds_TCs_r1_tracks, xds_TCs_r1_params = Extract_Circle(
    xds_wmo, pnt_lon, pnt_lat, r1, d_vns)

