#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op
import xarray as xr
import numpy as np

from lib.custom_plot import Plot_MJOphases

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
#times = DateConverter_Mat2Py(d_mjo['time'])
#
#ds_mjo = xr.Dataset(
#    {
#        'mjo':(('time',), d_mjo['mjo']),
#        'ph':(('time',), d_mjo['ph']),
#        'phase':(('time',), d_mjo['phase']),
#        'phi':(('time',), d_mjo['phi']),
#        'rmm1':(('time',), d_mjo['rmm1']),
#        'rmm2':(('time',), d_mjo['rmm2']),
#    },
#    {'time':times}
#)
#
#ds_mjo.to_netcdf(op.join(p_data, 'MJO.nc'),'w')


# -------------------------------------------------------------------
# Load MJO data and do analysis 

# Load a WeatherPredictor object from netcdf
p_mjo = op.join(p_data, 'MJO.nc')
ds_mjo = xr.open_dataset(p_mjo)

# select only data after initial year
y1 = '1979-01-01'
ds_mjo_cut = ds_mjo.loc[dict(time=slice(y1, None))]

# plot MJO data
#Plot_MJOphases(ds_mjo_cut)


# categorize MJO (25 types)
phase = ds_mjo_cut['phase']
rmm1 = ds_mjo_cut['rmm1']
rmm2 = ds_mjo_cut['rmm2']
rmm = np.sqrt(rmm1**2+rmm2**2)

# get category
categ = np.empty(rmm.shape)*np.nan
for i in range(1,9):
    s = np.squeeze(np.where(phase == i))
    rmm_p = rmm[s]

    # TODO: VA LENTO. OPTIMIZAR
    for j in s:
        if rmm[j] > 2.5:
            categ[j] = i
        elif rmm[j] > 1.5:
            categ[j] = i+8
        elif rmm[j] > 1:
            categ[j] = i+8*2
        elif rmm[j] <= 1:
            categ[j] = 25

print categ

import sys; sys.exit()
