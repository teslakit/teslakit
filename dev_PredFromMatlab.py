
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xarray as xr
import os.path as op

from lib.io.matlab import ReadMatfile
from lib.objs.predictor import WeatherPredictor as WP
from lib.custom_dateutils import DateConverter_Mat2Py

# data storage
p_data = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/'


# -------------------------------------------------------------------
# START FROM MATLAB DATA: CREATE TESLAKIT PREDICTOR

# predictor used: SST spatial fields (ERSST v4)
p_pred_mat = op.join(p_data, 'SST_1854_2017.mat')
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

# parse data to xr.DataArray
data_pred = xr.DataArray(var, coords=[lon, lat, time],
                         dims=['longitude', 'latitude', 'time'],
                         name=n_pred)

# cut bounding box
data_bb = data_pred.loc[lon1:lon2, lat1:lat2, :]

# Create a WeatherPredictor object and set data
p_pred_save = op.join(p_data, 'TKPRED_SST.nc') # file to save/load pred data
wpred = WP(p_pred_save)
wpred.SetData(data_bb)

# now save WeatherPredictor data to a netcdf
wpred.SaveData()

print wpred.data_set

