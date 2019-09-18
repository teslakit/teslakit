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
from teslakit.plotting.storms import Plot_TCs_TracksParams, \
Plot_Historical_TCs_Tracks, Plot_Historical_TCs_Tracks_Category


# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')


# Load storms extracted at 4 degree radius 
xds_tracks_r1, xds_params_r1 = db.Load_TCs_r1()
xds_tracks_r2, xds_params_r2 = db.Load_TCs_r2()


# Plot storms tracks and storm parametrized inside radius 
Plot_TCs_TracksParams(xds_tracks_r2, xds_params_r2)


# Plot storm tracks world map (requires basemap)
lon1, lon2 = 90, 270
lat1, lat2 = -20, 70
pnt_lon, pnt_lat = 167.5, 9.75
r1, r2 = 14, 4

Plot_Historical_TCs_Tracks(
    xds_tracks_r1, xds_tracks_r2,
    lon1, lon2, lat1, lat2,
    pnt_lon, pnt_lat, r1, r2,
)

# Plot storm tracks category world map (requires basemap)
category = xds_params_r1.category

Plot_Historical_TCs_Tracks_Category(
    xds_tracks_r1, category,
    lon1, lon2, lat1, lat2,
    pnt_lon, pnt_lat, r1,
)

