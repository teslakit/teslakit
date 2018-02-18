#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xarray as xr
import os.path as op

from lib.objs.predictor import WeatherPredictor as WP
from lib.custom_stats import ClassificationKMA
from lib.custom_plot import Plot_PredictorEOFs
from lib.objs.alr_enveloper import ALR_ENV


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

xds_AWT = ClassificationKMA(xds_pca, num_clusters, num_reps, repres)
print xds_AWT


## ----------------------------------
## Autoregressive Logistic Regression

bmus = xds_AWT['bmus'].values
t_data = xds_AWT['time']
num_wts = 6  # or len(set(bmus))

# Autoregressive logistic enveloper
ALRE = ALR_ENV(bmus, t_data, num_wts)

# ALR terms
d_terms_settings = {
    'mk_order'  : 1,
    'constant' : True,
    'time' : False,
    'seasonality': (False, []),
}

ALRE.SetFittingTerms(d_terms_settings)

# ALR model fitting
ALRE.FitModel()

# ALR model simulations 
sim_num = 10
sim_start = 1700
sim_end = 3200
sim_freq = '1y'

evbmus_sim, evbmus_probcum = ALRE.Simulate(sim_num, sim_start, sim_end,
                                           sim_freq)
print evbmus_sim
print evbmus_probcum

