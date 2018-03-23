#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import os.path as op
import numpy as np
import xarray as xr
from datetime import date, timedelta, datetime

from lib.mjo import GetMJOCategories, DownloadMJO
from lib.custom_plot import Plot_MJOphases, Plot_MJOCategories
from lib.objs.alr_enveloper import ALR_ENV

# data storage
p_data = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/'
p_mjo_hist = op.join(p_data, 'MJO_hist.nc')


# ---------------------------------
# Download mjo and mount xarray.dataset
#y1 = '1979-01-01'
#xds_mjo_hist = DownloadMJO(p_mjo_hist, init_year=y1)

# Load MJO data (previously downloaded)
xds_mjo_hist = xr.open_dataset(p_mjo_hist)


# ---------------------------------
# Calculate MJO categories (25 used) 
rmm1 = xds_mjo_hist['rmm1']
rmm2 = xds_mjo_hist['rmm2']
phase = xds_mjo_hist['phase']

categ, d_rmm_categ = GetMJOCategories(rmm1, rmm2, phase)
xds_mjo_hist['categ'] = (('time',), categ)


## plot MJO data
#Plot_MJOphases(rmm1, rmm2, phase)
#
## plot MJO categories
#Plot_MJOCategories(rmm1, rmm2, categ)



# ---------------------------------
# Autoregressive Logistic Regression

# MJO historical data for fitting
num_categs  = 25
xds_bmus_fit = xds_mjo_hist.categ


# Autoregressive logistic enveloper
ALRE = ALR_ENV(xds_bmus_fit, num_categs)

# ALR terms
d_terms_settings = {
    'mk_order'  : 3,
    'constant' : True,
    'seasonality': (True, [2,4,8]),
}

ALRE.SetFittingTerms(d_terms_settings)

# ALR model fitting
ALRE.FitModel(max_iter=10000)

# ALR model simulations 
sim_num = 1  # only one simulation for mjo daily
sim_years = 500

# simulation dates
d1 = date(1700,6,1)
d2 = date(d1.year+sim_years, d1.month, d1.day)
dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

# launch simulation
xds_alr = ALRE.Simulate(sim_num, dates_sim)
evbmus_sim = xds_alr.evbmus_sims.values

# parse to 1D array
evbmus_sim = np.squeeze(evbmus_sim)

# Generate mjo_sim list using random mjo from each category
# TODO: MUY LENTO, ACELERAR
mjo_sim = np.empty((len(evbmus_sim),2)) * np.nan
for c, m in enumerate(evbmus_sim):
    options = d_rmm_categ['cat_{0}'.format(int(m))]
    r = np.random.randint(options.shape[0])
    mjo_sim[c,:] = options[r,:]

# TODO COMO OBTENGO MJO SIMULATED PHASE?

# TODO: GUARDAR MJO SIMULADO, PENSAR ESTRUCTURA DE DATOS INTERNA PARA
# CONECTAR EL CODIGO en Project.py

p_mat_output = op.join(
    p_data, 'MJO_SIM_500y.mat')
with h5py.File(p_mat_output, 'w') as hf:
    hf['categ'] = evbmus_sim
    hf['dates'] = np.vstack(
        ([d.year for d in dates_sim],
        [d.month for d in dates_sim],
        [d.day for d in dates_sim])).T

