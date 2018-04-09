#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import xarray as xr
from datetime import datetime, timedelta

# tk libs
from lib.objs.predictor import WeatherPredictor as WP
from lib.custom_stats import ClassificationKMA
from lib.custom_plot import Plot_PredictorEOFs
from lib.objs.alr_wrapper import ALR_WRP

# data storage
p_data = op.join(op.dirname(__file__),'..','data')
p_export_figs = op.join(op.dirname(__file__),'..','data','export_figs')
p_pred = op.join(p_data, 'TKPRED_SST.nc')



# --------------------------------------
# Load a WeatherPredictor object from netcdf (parsed from .mat)
wpred = WP(p_pred)


# --------------------------------------
# Principal Components Analysis
y1 = 1880
yN = 2016
m1 = 6
mN = 5

xds_pca = wpred.CalcPCA(y1, yN, m1, mN)

# plot EOFs
#n_plot = 1
# TODO: INCORPORAR P_EXPORT
#Plot_PredictorEOFs(xds_pca, n_plot)


# --------------------------------------
# KMA Classification 
num_clusters = 6
num_reps = 2000
repres = 0.95

# TODO: ACABAR COPULAS DENTRO
xds_AWT = ClassificationKMA(
    xds_pca, num_clusters, num_reps, repres)

# TODO: GUARDAR xds_pca y xds_AWT PARA E1A_MMSL_KWA.m


# --------------------------------------
# Autoregressive Logistic Regression
xds_bmus_fit = xr.Dataset(
    {
        'bmus':(('time',), xds_AWT.bmus),
    },
    coords = {'time': xds_AWT.time.values}
).bmus

num_wts = 6
ALRW = ALR_WRP(xds_bmus_fit, num_wts)

# ALR terms
d_terms_settings = {
    'mk_order'  : 1,
    'constant' : True,
    'long_term' : False,
    'seasonality': (False, []),
}


ALRW.SetFittingTerms(d_terms_settings)

# ALR model fitting
ALRW.FitModel()

# ALR model simulations 
sim_num = 10
year_sim1 = 1700
year_sim2 = 2700

dates_sim = [
    datetime(x,1,1) for x in range(year_sim1,year_sim2+1)]

xds_alr = ALRW.Simulate(sim_num, dates_sim)

print xds_alr

