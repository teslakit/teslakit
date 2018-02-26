#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py # TODO temporal para exportar al final
import os.path as op
import xarray as xr
import pandas as pd
import numpy as np
from lib.objs.alr_enveloper import ALR_ENV
from lib.io.matlab import ReadMatfile as rmat
from lib.custom_dateutils import DateConverter_Mat2Py
from datetime import date, timedelta, datetime

# TODO: la gestion de datos temporales, datetime, np.datetime64 en este script
#es un desastre. Aprender a usar correctamente los tiempos de xarray.dataset

# TODO: 200 lineas solo para cargar y procesar datos, resumir 

# data storage
p_data = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/tests_ALR/'


## -------------------------------------------------------------------
## Get data used to FIT ALR model

## KMA: bmus
p_mat = op.join(p_data, 'KMA_daily_42.mat')
d_mat = rmat(p_mat)['KMA']
xds_KMA_fit = xr.Dataset(
    {
        'bmus':(('time',), d_mat['bmus']),
    },
    coords = {'time': [date(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)
# store datetime array
d1 = date(
    xds_KMA_fit.time.values[0].year,
    xds_KMA_fit.time.values[0].month,
    xds_KMA_fit.time.values[0].day
)
d2 = date(
    xds_KMA_fit.time.values[-1].year,
    xds_KMA_fit.time.values[-1].month,
    xds_KMA_fit.time.values[-1].day
)
dates_KMA_fit = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

## MJO: rmm1, rmm2 (first date 1979-01-01 in order to avoid nans)
p_mat = op.join(p_data, 'MJO.mat')
d_mat = rmat(p_mat)
xds_MJO_fit = xr.Dataset(
    {
        'rmm1': (('time',), d_mat['rmm1']),
        'rmm2': (('time',), d_mat['rmm2']),
    },
    coords = {'time': [date(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)
d1 = date(1979,01,01)  # reindex to 1979-01-01 forward
d2 = date(
    xds_MJO_fit.time.values[-1].year,
    xds_MJO_fit.time.values[-1].month,
    xds_MJO_fit.time.values[-1].day
)
rix = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]
xds_MJO_fit = xds_MJO_fit.reindex({'time':rix}, method='pad')
dates_MJO_fit = rix  # store datetime array


## AWT: PCs (annual data, parse to daily)
p_mat = op.join(p_data, 'PCs_for_AWT_mes10.mat')
d_mat = rmat(p_mat)['AWT']
xds_PCs_fit = xr.Dataset(
    {
        'PC1': (('time',), d_mat['PCs'][:,0]),
        'PC2': (('time',), d_mat['PCs'][:,1]),
        'PC3': (('time',), d_mat['PCs'][:,2]),
    },
    coords = {'time': [date(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)
# reindex annual data to daily data
# TODO: SE REPITE MUCHO, BUSCAR FORMA DIRECTA DE PARSEAR A DATETIME O METERLO
# EN FUNCION
d1 = date(
    xds_PCs_fit.time.values[0].year,
    xds_PCs_fit.time.values[0].month,
    xds_PCs_fit.time.values[0].day
)
d2 = date(
    xds_PCs_fit.time.values[-1].year,
    xds_PCs_fit.time.values[-1].month,
    xds_PCs_fit.time.values[-1].day
)
rix = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]
xds_PCs_fit = xds_PCs_fit.reindex({'time':rix}, method='pad')
dates_PCs_fit = rix  # store datetime array



## -------------------------------------------------------------------
## Get data used to SIMULATE with trained ALR model 

## MJO: rmm1, rmm2 (daily data)
p_mat = op.join(p_data, 'MJO_500_part1.mat')
#p_mat = op.join(p_data, '?????????')  # TODO: usar 1000y
d_mat = rmat(p_mat)
xds_MJO_sim = xr.Dataset(
    {
        'rmm1': (('time',), d_mat['rmm1']),
        'rmm2': (('time',), d_mat['rmm2']),
    },
    coords = {'time': [date(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)
# store datetime array
d1 = date(
    xds_MJO_sim.time.values[0].year,
    xds_MJO_sim.time.values[0].month,
    xds_MJO_sim.time.values[0].day
)
d2 = date(
    xds_MJO_sim.time.values[-1].year,
    xds_MJO_sim.time.values[-1].month,
    xds_MJO_sim.time.values[-1].day
)
dates_MJO_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]


## AWT: PCs (annual data, parse to daily)
p_mat = op.join(p_data, 'AWT_PCs_500_part1.mat')
#p_mat = op.join(p_data, 'AWT_forALR.mat')  # TODO: usar 1000y
d_mat = rmat(p_mat)['AWT']
xds_PCs_sim = xr.Dataset(
    {
        'PC1': (('time',), d_mat['PCs'][:,0]),
        'PC2': (('time',), d_mat['PCs'][:,1]),
        'PC3': (('time',), d_mat['PCs'][:,2]),
    },
    coords = {'time': [date(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)
# reindex annual data to daily data
d1 = date(
    xds_PCs_sim.time.values[0].year,
    xds_PCs_sim.time.values[0].month,
    xds_PCs_sim.time.values[0].day
)
d2 = date(
    xds_PCs_sim.time.values[-1].year,
    xds_PCs_sim.time.values[-1].month,
    xds_PCs_sim.time.values[-1].day
)
rix = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]
xds_PCs_sim = xds_PCs_sim.reindex({'time':rix}, method='pad')
dates_PCs_sim = rix  # store datetime array



## -------------------------------------------------------------------
## Mount covariates matrix

# available data
# model fit: xds_KMA_fit, xds_MJO_fit, xds_PCs_fit
# model sim: xds_MJO_sim, xds_PCs_sim


# covariates: FIT
d1 = max(dates_MJO_fit[0], dates_PCs_fit[0])
d2 = min(dates_MJO_fit[-1], dates_PCs_fit[-1])
dates_covar_fit = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

# PCs covar 
cov_PCs = xds_PCs_fit.sel(time=slice(dates_covar_fit[0],dates_covar_fit[-1]))
cov_1 = cov_PCs.PC1.values.reshape(-1,1)
cov_2 = cov_PCs.PC2.values.reshape(-1,1)
cov_3 = cov_PCs.PC3.values.reshape(-1,1)

# MJO covars
cov_MJO = xds_MJO_fit.sel(time=slice(dates_covar_fit[0],dates_covar_fit[-1]))
cov_4 = cov_MJO.rmm1.values.reshape(-1,1)
cov_5 = cov_MJO.rmm2.values.reshape(-1,1)

# join covars and norm.
cov_T = np.hstack((cov_1, cov_2, cov_3, cov_4, cov_5))

# KMA related covars starting at KMA period 
i0 = dates_covar_fit.index(xds_KMA_fit.time.values[0])
cov_KMA = cov_T[i0:,:]

# normalize
cov_norm_fit = (cov_KMA - cov_T.mean(axis=0)) / cov_T.std(axis=0)


# covariates: SIMULATION
d1 = max(dates_MJO_sim[0], dates_PCs_sim[0])
d2 = min(dates_MJO_sim[-1], dates_PCs_sim[-1])
dates_covar_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

# PCs covar 
cov_PCs = xds_PCs_sim.sel(time=slice(dates_covar_sim[0],dates_covar_sim[-1]))
cov_1 = cov_PCs.PC1.values.reshape(-1,1)
cov_2 = cov_PCs.PC2.values.reshape(-1,1)
cov_3 = cov_PCs.PC3.values.reshape(-1,1)

# MJO covars
cov_MJO = xds_MJO_sim.sel(time=slice(dates_covar_sim[0],dates_covar_sim[-1]))
cov_4 = cov_MJO.rmm1.values.reshape(-1,1)
cov_5 = cov_MJO.rmm2.values.reshape(-1,1)

# join covars (do not normalize simulation covariates)
cov_T_sim = np.hstack((cov_1, cov_2, cov_3, cov_4, cov_5))




## -------------------------------------------------------------------
## Autoregressive Logistic Regression


# use bmus inside covariate time frame
i0 = dates_KMA_fit.index(max(dates_covar_fit[0], dates_KMA_fit[0]))
i1 = dates_KMA_fit.index(min(dates_covar_fit[-1], dates_KMA_fit[-1]))+1

bmus = xds_KMA_fit.bmus[i0:i1]
t_data = bmus.time.values
num_clusters  = 42


# Autoregressive logistic enveloper
ALRE = ALR_ENV(bmus, t_data, num_clusters)

# ALR terms
d_terms_settings = {
    'mk_order'  : 3,
    'constant' : True,
    'long_term' : False,
    'seasonality': (True, [2]),
    'covariates': (True, cov_norm),
}

ALRE.SetFittingTerms(d_terms_settings)


# ALR fit model
ALRE.FitModel()

# save ALR for future simulations
ALRE.SaveModel(op.join(p_data, 'ALR_model_t1_allterms.sav'))


# ALR model simulations 
sim_num = 5
sim_start = 1700
sim_end = 1705
sim_freq = '1d'

# launch simulation
evbmus_sim, evbmus_probcum, dates_sim = ALRE.Simulate(
    sim_num, sim_start, sim_end, sim_freq, cov_T_sim)


# Save results for matlab plot 
p_mat_output = op.join(p_data, 'alrout_t1_allterms_5y5s.h5')
with h5py.File(p_mat_output, 'w') as hf:
    hf['bmusim'] = evbmus_sim
    hf['probcum'] = evbmus_probcum
    hf['dates'] = np.vstack(
        ([d.year for d in dates_sim],
        [d.month for d in dates_sim],
        [d.day for d in dates_sim])).T

