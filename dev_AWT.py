#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xarray as xr
import os.path as op

from lib.objs.predictor import WeatherPredictor as WP
from lib.custom_stats import ClassificationKMA
from lib.alr import AutoRegLogisticReg
from lib.custom_plot import Plot_PredictorEOFs


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
num_wts = 6  # or len(set(bmus))
num_sims = 100
sim_start = 1700
sim_end = 3701
mk_order = 1

# Autoregressive terms
d_ALR_terms = {
    'mk_order'  : (True, mk_order),
    'constant_term' : (True,),
    'time_term' : (False, ),
    'seasonality_term': (False,[],[]),
}

evbmusd_sim = AutoRegLogisticReg(
    bmus, num_wts, num_sims, sim_start, sim_end)

print evbmusd_sim

