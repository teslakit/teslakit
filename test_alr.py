#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op
import xarray as xr
from lib.objs.alr_enveloper import ALR_ENV

# data storage
p_data = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/'

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
sim_end = 2402
sim_freq = '1d'

evbmus_sim, evbmus_probcum = ALRE.Simulate(sim_num, sim_start, sim_end,
                                           sim_freq)

