#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import xarray as xr
import numpy as np

# tk libs
from lib.objs.tkpaths import PathControl
from lib.data_fetcher.STORMS import Download_NOAA_WMO
from lib.tcyclone import Extract_Circle
from lib.plotting.storms import WorldMap_Storms


# --------------------------------------
# data storage and path control
pc = PathControl()
pc.SetSite('test_site')


# --------------------------------------
# Historical TCs  

# Download TCs and save xarray.dataset to netcdf
download = False
if download:
    xds_wmo = Download_NOAA_WMO(pc.DB.tcs.noaa)
else:
    xds_wmo = xr.open_dataset(pc.DB.tcs.noaa)

# set lon to 0-360
lon_wmo = xds_wmo.lon_wmo.values[:]
lon_wmo[np.where(lon_wmo<0)] = lon_wmo[np.where(lon_wmo<0)]+360
xds_wmo['lon_wmo'].values[:] = lon_wmo

# modify some variable names
xds_wmo.rename(
    {'lon_wmo':'lon',
     'lat_wmo':'lat',
     'time_wmo':'dates',
     'pres_wmo':'pressure',
    }, inplace=True)

# store fixed wmo file
xds_wmo.to_netcdf(pc.DB.tcs.noaa_fix)


# Select TCs that crosses a circular area 
p_lon = 178
p_lat = -17.5
r = 4

xds_TCs_r, xds_inside = Extract_Circle(
    xds_wmo, p_lon, p_lat, r)

# store data
xds_TCs_r.to_netcdf(pc.site.tcs.circle_hist)


# TODO: ADD PLOT (trazas huracanes seleccionados sobre mapa mundo)
#WorldMap_Storms(xds_hurr_r)

