#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op
import xarray as xr


# data storage
p_data = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/'


# -------------------------------------------------------------------
## parse MJO.mat to netcdf dataset
#from lib.io.matlab import ReadMatfile as rmat
#from lib.custom_dateutils import DateConverter_Mat2Py
#
#p_mjo_mat = op.join(p_data, 'MJO.mat')
#d_mjo = rmat(p_mjo_mat)
#dim_mjo = len(d_mjo['mjo'])
#
#ds_mjo = xr.Dataset(
#    {
#        'mjo':(('time',), d_mjo['mjo']),
#        'ph':(('time',), d_mjo['ph']),
#        'phase':(('time',), d_mjo['phase']),
#        'phi':(('time',), d_mjo['phi']),
#        'rmm1':(('time',), d_mjo['rmm1']),
#        'rmm2':(('time',), d_mjo['rmm2']),
#        'time':(('time',), DateConverter_Mat2Py(d_mjo['time']))
#    }
#)
#
#ds_mjo.to_netcdf(op.join(p_data, 'MJO.nc'),'w')


# -------------------------------------------------------------------
# LOAD TESLAKIT PREDICTOR AND DO PRINCIPAL COMPONENTS ANALYSIS

# Load a WeatherPredictor object from netcdf
p_mjo = op.join(p_data, 'MJO.nc')

ds_mjo = xr.open_dataset(p_mjo)
print ds_mjo
