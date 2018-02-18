#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op
import xarray as xr

from lib.mjo import GetMJOCategories
from lib.custom_plot import Plot_MJOphases, Plot_MJOCategories
from lib.objs.alr_enveloper import ALR_ENV

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
# Load a MJO data from netcdf
p_mjo = op.join(p_data, 'MJO.nc')
ds_mjo = xr.open_dataset(p_mjo)

# select only data after initial year
y1 = '1979-01-01'
ds_mjo_cut = ds_mjo.loc[dict(time=slice(y1, None))]

# set MJO categories (25)
rmm1 = ds_mjo_cut['rmm1']
rmm2 = ds_mjo_cut['rmm2']
phase = ds_mjo_cut['phase']

categ, d_rmm_categ = GetMJOCategories(rmm1, rmm2, phase)

# add categories to MJO Dataset and save
ds_mjo_cut['categ'] = (('time',), categ)

# save dataset
p_mjo_cut = op.join(p_data, 'MJO_categ.nc')
ds_mjo_cut.to_netcdf(p_mjo_cut)


# plot MJO data
#Plot_MJOphases(ds_mjo_cut)

# plot MJO categories
#Plot_MJOCategories(ds_mjo_cut)


# -----------------------------------
# Autoregressive Logistic Regression

# Load a MJO data from netcdf
p_mjo_cut = op.join(p_data, 'MJO_categ.nc')
ds_mjo_cut = xr.open_dataset(p_mjo_cut)

bmus = ds_mjo_cut['categ'].values
t_data = ds_mjo_cut['time']
num_categs  = 25

# Autoregressive logistic enveloper
ALRE = ALR_ENV(bmus, t_data, num_categs)

# ALR terms
d_terms_settings = {
    'mk_order'  : 3,
    'constant' : True,
    'time' : False,
    'seasonality': (True, [2,4,8]),
}

ALRE.SetFittingTerms(d_terms_settings)

# ALR model fitting
ALRE.FitModel()

# ALR model simulations 
sim_num = 1
sim_start = 1900
sim_end = 2602
sim_freq = '1d'

evbmus_sim, evbmus_probcum = ALRE.Simulate(sim_num, sim_start, sim_end,
                                           sim_freq)

print evbmus_sim
print evbmus_probcum

