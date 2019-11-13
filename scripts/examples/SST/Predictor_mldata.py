#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op

# pip
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../../')

# teslakit
from teslakit.project_site import PathControl
from teslakit.io.matlab import ReadMatfile
from teslakit.pca import CalcRunningMean
from teslakit.util.time_operations import DateConverter_Mat2Py

# --------------------------------------
# data storage and path control
pc = PathControl()

# TODO: revisar datos

# data storage
p_pred_mat = op.join(pc.p_DB, 'SST', 'SST_1854_2017.mat')
p_pred_nc = op.join(pc.p_DB, 'SST', 'SST_1854_2017_Pacific.nc')


# --------------------------------------
# Parse predictor from old .mat to netcdf 

# predictor used: SST spatial fields (ERSST v4)
n_pred = 'SST';

# load data (from matlab file)
d_matf = ReadMatfile(p_pred_mat)
d_pred = d_matf[n_pred]
lat = d_pred['lat']
lon = d_pred['lon']
var = d_pred['sst']
time = DateConverter_Mat2Py(d_pred['time'])  # matlab datenum to python datetime


# parse data to xr.Dataset
xds_predictor = xr.Dataset(
    {
        'SST': (('longitude','latitude','time'), var),
    },
    coords = {
        'longitude': lon,
        'latitude': lat,
        'time': time
    }
)

# cut bounding box
lat1, lat2 = 5, -5
lon1, lon2 = 120, 280
xds_predictor = xds_predictor.sel(
    longitude=slice(lon1,lon2),
    latitude=slice(lat1,lat2)
)


# calculate running average grouped by months 
xds_predictor = CalcRunningMean(xds_predictor, 'SST', 5)

# save netcdf
xds_predictor.to_netcdf(p_pred_nc, 'w')

