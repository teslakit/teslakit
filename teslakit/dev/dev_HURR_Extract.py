#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import xarray as xr
import numpy as np

# tk libs
from lib.data_fetcher.HURRICANES import Download_HURRS
from lib.hurricanes import Extract_Circle

# data storage
p_data = op.join(op.dirname(__file__), '..', 'data')
p_data_hurr = op.join(p_data, 'HURR')
p_hurr_noaa = op.join(p_data_hurr, 'Allstorms.ibtracs_wmo.v03r10.nc')

p_hurr_area = op.join(p_data_hurr, 'storms.ibtracs_wmo.v03r10_circle.nc')



# Download hurricanes and save xarray.dataset to netcdf
download = False
if download:
    xds_wmo = Download_HURRS(p_data_hurr)
else:
    xds_wmo = xr.open_dataset(p_hurr_noaa)

# set lon to 0-360
lon_wmo = xds_wmo.lon_wmo.values[:]
lon_wmo[np.where(lon_wmo<0)] = lon_wmo[np.where(lon_wmo<0)]+360
xds_wmo['lon_wmo'].values[:] = lon_wmo


# Select hurricanes that crosses a circular area 
p_lon = 178
p_lat = -17.5
r = 4  # degree

xds_hurr_area = Extract_Circle(
    xds_wmo, p_lon, p_lat, r)
xds_hurr_area.to_netcdf(p_hurr_area)

