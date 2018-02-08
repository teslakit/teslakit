#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op
import xarray as xr

from lib.mjo import GetMJOCategories
from lib.custom_plot import Plot_MJOphases, Plot_MJOCategories
from lib.alr import AutoRegLogisticReg

# data storage
p_data = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/'


# -------------------------------------------------------------------
## parse MJO.mat to netcdf dataset
#
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
# Load MJO data and do categories analisys 
#
## Load a MJO data from netcdf
#p_mjo = op.join(p_data, 'MJO.nc')
#ds_mjo = xr.open_dataset(p_mjo)
#
## select only data after initial year
#y1 = '1979-01-01'
#ds_mjo_cut = ds_mjo.loc[dict(time=slice(y1, None))]
#
## set MJO categories (25)
#rmm1 = ds_mjo_cut['rmm1']
#rmm2 = ds_mjo_cut['rmm2']
#phase = ds_mjo_cut['phase']
#
#categ, d_rmm_categ = GetMJOCategories(rmm1, rmm2, phase)
#
## add categories to MJO Dataset and save
#ds_mjo_cut['categ'] = (('time',), categ)
#
## save dataset
#ds_mjo_cut.to_netcdf(op.join(p_data, 'MJO_categ.nc'),'w')


# plot MJO data
#Plot_MJOphases(ds_mjo_cut)

# plot MJO categories
#Plot_MJOCategories(ds_mjo_cut)


# -----------------------------------
## Autoregressive Logistic Regression

# Load a MJO data from netcdf
p_mjo_cut = op.join(p_data, 'MJO_categ.nc')
ds_mjo_cut = xr.open_dataset(p_mjo_cut)


# TODO: VERSIONAR EL num_years
from lib.alr import num_years
import numpy as np
from datetime import datetime
time_data = ds_mjo_cut['time']

y1 = time_data[0].dt.year
m1 = time_data[0].dt.month
d1 = time_data[0].dt.day
y2 = time_data[-1].dt.year
m2 = time_data[-1].dt.month
d2 = time_data[-1].dt.day


# TODO: CREO que esto devuelve el numero de anios entre dos fechas..
# TODO: ESTO DEBERIA SER UNA FUNCION PROPIA EN PYTHON, REEMPLAZAR
print num_years(
    datetime(y1,m1,d1),
    datetime(y2,m2,d2))

import sys
sys.exit()

t = np.zeros_like(time_data, dtype=np.float)

for i in range(len(time_data)):
    t[i] = num_years(
        datetime(time_data[0].year,
            time_data[0].month,
            time_data[0].day),
        datetime(time_data[i].year,
            time_data[i].month,
            time_data[i].day))


print t








import sys; sys.exit()

bmus = ds_mjo['categ'].values
t_data = ds_mjo['time'].values
num_wts = 25
num_sims = 1
sim_start = 1700
sim_end = 2402
mk_order = 2

evbmusd_sim = AutoRegLogisticReg(
    bmus, num_wts, num_sims, sim_start, sim_end,
    mk_order=2, time_data=t_data)

print evbmusd_sim

