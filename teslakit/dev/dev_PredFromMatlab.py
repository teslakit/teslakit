#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import xarray as xr

# tk libs
from lib.io.matlab import ReadMatfile
from lib.predictor import CalcRunningMean
from lib.custom_dateutils import DateConverter_Mat2Py

# data storage
p_data = op.join(op.dirname(__file__),'..','data')
p_pred_mat = op.join(p_data, 'SST_1854_2017.mat')
p_pred_nc = op.join(p_data, 'SST_1985_2017.nc')


# --------------------------------------
# Parse predictor from old .mat to netcdf 

# predictor used: SST spatial fields (ERSST v4)
n_pred = 'SST';
lat1, lat2 = 5, -5
lon1, lon2 = 120, 280

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
xds_predictor = xds_predictor.sel(
    longitude=slice(lon1,lon2),
    latitude=slice(lat1,lat2)
)


# calculate running average grouped by months 
xds_predictor = CalcRunningMean(xds_predictor, 'SST', 5)

# save netcdf
xds_predictor.to_netcdf(p_pred_nc, 'w')

print xds_predictor

