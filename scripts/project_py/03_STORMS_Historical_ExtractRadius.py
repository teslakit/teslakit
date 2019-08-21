#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op

# pip
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [18, 8]

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../')

# teslakit
from teslakit.project_site import Site
from teslakit.storms import Extract_Circle
from teslakit.plotting.storms import WorldMap_Storms
from teslakit.statistical import CopulaSimulation


# --------------------------------------
# Site paths and parameters
data_folder = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
site = Site(data_folder, 'KWAJALEIN_TEST')

ST = site.pc.site                          # site database
PR = site.params                           # site parameters

# input files: NOAA WMO TCs
p_hist_tcs = op.join(data_folder, 'database', 'TCs', 'Allstorms.ibtracs_wmo.v03r10.nc')

# output files
p_hist_r1 = ST.TCs.hist_r1                 # historical TCs inside radius 1
p_hist_r1_params = ST.TCs.hist_r1_params   # TCs parameters inside radius 1
p_hist_r2 = ST.TCs.hist_r2                 # historical TCs inside radius 2
p_hist_r2_params = ST.TCs.hist_r2_params   # TCs parameters inside radius 2

# wave point lon, lat and radius for TCs selection
pnt_lon = float(PR.WAVES.point_longitude)
pnt_lat = float(PR.WAVES.point_latitude)
r1 = float(PR.TCS.r1)   # bigger one
r2 = float(PR.TCS.r2)   # smaller one


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


# Select TCs that crosses a circular area R1
print(
'\nExtracting Historical TCs from WMO database...\n\
    Lon = {0:.2f}º , Lat = {1:.2f}º, R = {2:6.2f}º'.format(
    pnt_lon, pnt_lat, r1)
)

xds_TCs_r1, xds_TCs_r1_params = Extract_Circle(
    xds_wmo, pnt_lon, pnt_lat, r1, d_vns)

# store data
xds_TCs_r1.to_netcdf(p_hist_r1)
xds_TCs_r1_params.to_netcdf(p_hist_r1_params)

print('\nHistorical TCs selection and parameters stored at:\n{0}\n{1}'.format(
    p_hist_r1, p_hist_r1_params))


# Select TCs that crosses a circular area R2
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

