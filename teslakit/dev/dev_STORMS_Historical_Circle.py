#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import xarray as xr
import numpy as np

# tk libs
from lib.objs.tkpaths import Site
from lib.data_fetcher.STORMS import Download_NOAA_WMO
from lib.tcyclone import Extract_Circle
from lib.plotting.storms import WorldMap_Storms


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')

DB = site.pc.DB                        # common database
ST = site.pc.site                      # site database
PR = site.params                       # site parameters

# input files
p_wvs_parts = ST.wvs.partitions_p1
p_hist_tcs = DB.tcs.noaa_fix

# output files
p_hist_tcs = DB.tcs.noaa
p_hist_tcs_fix = DB.tcs.noaa_fix
p_tcs_circle_hist = ST.tcs.circle_hist

# wave point lon, lat and radius for TCs selection
pnt_lon = float(PR.WAVES.point_longitude)
pnt_lat = float(PR.WAVES.point_latitude)
r2 = float(PR.TCS.r2)   # smaller one


# --------------------------------------
# Historical TCs  

# Download TCs and save xarray.dataset to netcdf
download = False
if download:
    xds_wmo = Download_NOAA_WMO(p_hist_tcs)
else:
    xds_wmo = xr.open_dataset(p_hist_tcs)

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
xds_wmo.to_netcdf(p_hist_tcs_fix)


# Select TCs that crosses a circular area 
print(
'\nExtracting Historical TCs from WMO database...\n \
Lon = {0:.2f}º , Lat = {1:.2f}º, R2  = {2:6.2f}º'.format(
    pnt_lon, pnt_lat, r2)
)

xds_TCs_r, xds_inside = Extract_Circle(
    xds_wmo, pnt_lon, pnt_lat, r2)

# store data
xds_TCs_r.to_netcdf(p_tcs_circle_hist)
print('\nHistorical TCs (inside circle) stored at:\n{0}'.format(p_tcs_circle_hist))


# TODO: ADD PLOT (trazas huracanes seleccionados sobre mapa mundo)
#WorldMap_Storms(xds_hurr_r)

