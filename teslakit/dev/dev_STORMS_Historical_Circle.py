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
p_hist_tcs = DB.TCs.noaa

# output files
p_hist_r1 = ST.TCs.hist_r1
p_hist_r1_params = ST.TCs.hist_r1_params
p_hist_r2 = ST.TCs.hist_r2
p_hist_r2_params = ST.TCs.hist_r2_params

# wave point lon, lat and radius for TCs selection
pnt_lon = float(PR.WAVES.point_longitude)
pnt_lat = float(PR.WAVES.point_latitude)
r1 = float(PR.TCS.r1)   # bigger one
r2 = float(PR.TCS.r2)   # smaller one

# TODO
# EL PROCESO RADIUS -> PARAMS -> COPULAS PARETO -> MDA -> RBF
# ES PARA LOS STORMS EN R1 (14º) O R2 (4º) 


# --------------------------------------
# Select Historical TCs inside circle

# Load historical TCs 
xds_wmo = xr.open_dataset(p_hist_tcs)

# dictionary with needed variable names 
d_vns = {
    'longitude': 'lon_wmo',
    'latitude': 'lat_wmo',
    'time': 'time_wmo',
    'pressure': 'pres_wmo',
}

# Select TCs that crosses a circular area R
print(
'\nExtracting Historical TCs from WMO database...\n\
    Lon = {0:.2f}º , Lat = {1:.2f}º, R = {2:6.2f}º'.format(
    pnt_lon, pnt_lat, r2)
)

xds_TCs_r2, xds_TCs_r2_params = Extract_Circle(
    xds_wmo, pnt_lon, pnt_lat, r2, d_vns)

# store data
xds_TCs_r2.to_netcdf(p_hist_r2)
xds_TCs_r2_params.to_netcdf(p_hist_r2_params)

print('\nHistorical TCs selection and parameters stored at:\n{0}\n{1}'.format(
    p_hist_r2, p_hist_r2_params))

