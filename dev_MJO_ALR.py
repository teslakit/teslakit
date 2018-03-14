#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op
import numpy as np
import xarray as xr
from datetime import date, timedelta, datetime

from lib.mjo import GetMJOCategories, DownloadMJO
from lib.custom_plot import Plot_MJOphases, Plot_MJOCategories
from lib.objs.alr_enveloper import ALR_ENV

# data storage
p_data = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/'

p_mjo_hist = op.join(p_data, 'historical', 'MJO_hist.nc')


# ---------------------------------
# Download mjo and mount xarray.dataset
#y1 = '1979-01-01'
#ds_mjo_hist = DownloadMJO(p_mjo_hist, init_year=y1)

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
ALRE.FitModel()

# ALR model simulations 
sim_num = 1  # only one simulation for mjo daily
sim_years = 15

# simulation dates
d1 = date(1900,1,1)
d2 = date(d1.year+sim_years, d1.month, d1.day)
dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

# launch simulation
evbmus_sim, _ = ALRE.Simulate(
    sim_num, dates_sim)

# parse to 1D array
evbmus_sim = np.squeeze(evbmus_sim)

# Generate mjo_sim list using random mjo from each category
mjo_sim = np.empty((len(evbmus_sim),2)) * np.nan
for c, m in enumerate(evbmus_sim):
    options = d_rmm_categ['cat_{0}'.format(int(m))]
    r = np.random.randint(options.shape[0])
    mjo_sim[c,:] = options[r,:]

# TODO: GUARDAR MJO SIMULADO, PENSAR ESTRUCTURA DE DATOS INTERNA PARA
# CONECTAR EL CODIGO
print mjo_sim

