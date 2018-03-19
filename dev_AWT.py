#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xarray as xr
import os.path as op

from lib.objs.predictor import WeatherPredictor as WP
from lib.custom_stats import ClassificationKMA
from lib.custom_plot import Plot_PredictorEOFs
from lib.objs.alr_enveloper import ALR_ENV

from datetime import datetime, timedelta


# data storage
p_data = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/'


# -------------------------------------------------------------------
# LOAD TESLAKIT PREDICTOR AND DO PRINCIPAL COMPONENTS ANALYSIS

# Load a WeatherPredictor object from netcdf
p_pred_save = op.join(p_data, 'TKPRED_SST.nc')
wpred = WP(p_pred_save)

# calculate running average grouped by months and save
#wpred.CalcRunningMean(5)
#wpred.SaveData()


## ----------------------------------
# Principal Components Analysis

y1 = 1880
yN = 2016
m1 = 6
mN = 5

xds_pca = wpred.CalcPCA(y1, yN, m1, mN)
print xds_pca

# plot EOFs
#n_plot = 1
#Plot_PredictorEOFs(xds_pca, n_plot)


## ----------------------------------
# KMA Classification 

num_clusters = 6
num_reps = 2000
repres = 0.95

# TODO: ACABAR COPULAS DENTRO
xds_AWT = ClassificationKMA(
    xds_pca, num_clusters, num_reps, repres)



## ----------------------------------
## Autoregressive Logistic Regression

xds_bmus_fit = xr.Dataset(
    {
        'bmus':(('time',), xds_AWT.bmus),
    },
    coords = {'time': xds_AWT.time.values}
).bmus

num_wts = 6
ALRE = ALR_ENV(xds_bmus_fit, num_wts)

# ALR terms
d_terms_settings = {
    'mk_order'  : 1,
    'constant' : True,
    'long_term' : False,
    'seasonality': (False, []),
}

ALRE.SetFittingTerms(d_terms_settings)

# ALR model fitting
ALRE.FitModel()

# ALR model simulations 
sim_num = 10
year_sim1 = 1700
year_sim2 = 2700

dates_sim = [
    datetime(x,1,1) for x in range(year_sim1,year_sim2+1)]

evbmus_sim, evbmus_probcum = ALRE.Simulate(
    sim_num, dates_sim)

print evbmus_sim
print evbmus_probcum

