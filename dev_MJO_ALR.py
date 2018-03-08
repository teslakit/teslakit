#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op
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
    'constant' : True,
    'seasonality': (True, [2,4,8]),
}

ALRE.SetFittingTerms(d_terms_settings)

# ALR model fitting
ALRE.FitModel()

# ALR model simulations 
sim_num = 4
sim_years = 10

# simulation dates
d1 = date(1900,1,1)
d2 = date(d1.year+sim_years, d1.month, d1.day)
dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]


# print some info
print 'ALR model fitted with data: {0} --- {1}'.format(
    xds_bmus_fit.time.values[0], xds_bmus_fit.time.values[-1])
print 'ALR model simulations with data: {0} --- {1}'.format(
    dates_sim[0], dates_sim[-1])


# launch simulation
evbmus_sim, evbmus_probcum = ALRE.Simulate(
    sim_num, dates_sim)

print evbmus_sim
print evbmus_probcum


# TODO: AHORA TENGO QUE PASAR EVBMUSSIM A UNA SERIE DE DATOS RMM1.RMM2 
# GENERADA "PSEUDOALEATORIAMENTE" "CAYENDO EN LAS CATEGORIAS Y COGIENDO UNO AL
# AZAR" DWT_PREPARE_MJO_AWT_2P
