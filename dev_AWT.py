#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xarray as xr

from lib.io.matlab import ReadMatfile
from lib.objs.predictor import WeatherPredictor as WP
from lib.custom_dateutils import DateConverter_Mat2Py
from lib.custom_stats import Classification_KMA

# predictor used: SST spatial fields (ERSST v4)
p_pred = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/SST_1854_2017.mat'
n_pred = 'SST';
lat1, lat2 = 5, -5
lon1, lon2 = 120, 280

# load data (from matlab file)
d_matf = ReadMatfile(p_pred)
d_pred = d_matf[n_pred]
lat = d_pred['lat']
lon = d_pred['lon']
var = d_pred['sst']
time = DateConverter_Mat2Py(d_pred['time'])  # matlab datenum to python datetime

# parse data to xr.DataArray
data_pred = xr.DataArray(var, coords=[lon, lat, time],
                         dims=['longitude', 'latitude', 'time'])

# cut bounding box
data_bb = data_pred.loc[lon1:lon2, lat1:lat2, :]


# Create a WeatherPredictor object
wpred = WP(data_bb, n_pred)

# calculate running average grouped by months
wpred.CalcRunningMean(5)


# ---------------------------------------------------------------------------
# Principal Components Analysis
y1 = 1880
yN = 2016
m1 = 6
mN = 5

wpred.CalcPCA(y1, yN, m1, mN)


# ---------------------------------------------------------------------------
# Principal Components Analysis
num_clusters = 6
num_reps = 2000
repres = 0.95

AWT = ClassificationKMA(wpred.PCA, num_clusters, num_reps, repres)


# ---------------------------------------------------------------------------
# TODO: GUARDAR CALCULOS REALIZADOS 
# TODO: CONTINUAR CON LA REGRESION LOGISTICA
# TODO: PLOTEOS
