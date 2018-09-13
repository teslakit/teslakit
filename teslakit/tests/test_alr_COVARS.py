#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import xarray as xr
import numpy as np
from datetime import datetime, timedelta

# tk libs
from lib.objs.alr_wrapper import ALR_WRP
from lib.io.matlab import ReadMatfile as rmat
from lib.custom_dateutils import xds2datetime as x2d
from lib.custom_dateutils import xds_reindex_daily as xr_daily
from lib.custom_dateutils import xds_common_dates_daily as xcd_daily


# data storage
p_data = op.join(op.dirname(__file__),'..','data')
p_data = op.join(p_data, 'tests', 'tests_ALR', 'tests_ALR_statsmodel')


# TODO CON TAIRUA MEJORAMOS EL CODIGO, COMPROBAR COMPATIBILIDAD


# --------------------------------------
# Get data used to FIT ALR model and preprocess

# KMA: bmus
# TODO: ESTOS SALEN DE UN PREPROCESO (ESTELA + TCS)
p_mat = op.join(p_data, 'KMA_daily_42.mat')
d_mat = rmat(p_mat)['KMA']
xds_KMA_fit = xr.Dataset(
    {
        'bmus':(('time',), d_mat['bmus']),
    },
    coords = {'time': [datetime(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)


# MJO: rmm1, rmm2 (first date 1979-01-01 in order to avoid nans)
# TODO: ESTOS DATOS SON HISTORICOS (COVARIATES).
# TODO: CAMBIAR A LEER DE LA BASE EN NETCDF
p_mat = op.join(p_data, 'MJO.mat')
d_mat = rmat(p_mat)
xds_MJO_fit = xr.Dataset(
    {
        'rmm1': (('time',), d_mat['rmm1']),
        'rmm2': (('time',), d_mat['rmm2']),
    },
    coords = {'time': [datetime(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)
# reindex to daily data after 1979-01-01 (avoid NaN) 
xds_MJO_fit = xr_daily(xds_MJO_fit, datetime(1979,01,01))


# AWT: PCs (annual data, parse to daily)
# TODO: VIENE DE DEV_AWT / PREDICTOR.CALCPCA
p_mat = op.join(p_data, 'PCs_for_AWT.mat')
d_mat = rmat(p_mat)['AWT']
xds_PCs_fit = xr.Dataset(
    {
        'PC1': (('time',), d_mat['PCs'][:,0]),
        'PC2': (('time',), d_mat['PCs'][:,1]),
        'PC3': (('time',), d_mat['PCs'][:,2]),
    },
    coords = {'time': [datetime(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)
# reindex annual data to daily data
xds_PCs_fit = xr_daily(xds_PCs_fit)



# --------------------------------------
# Get data used to SIMULATE with ALR model 
# TODO: ESTOS DATOS SON LOS QUE VIENEN DEL PROCESO PREVIO (DEV_X)

# MJO: rmm1, rmm2 (daily data)
# TODO: A PARTIR DE EVBMUS_SIM EN DEV_MJO_ALR 
p_mat = op.join(p_data, 'MJO_500_part1.mat')
d_mat = rmat(p_mat)
xds_MJO_sim = xr.Dataset(
    {
        'rmm1': (('time',), d_mat['rmm1']),
        'rmm2': (('time',), d_mat['rmm2']),
    },
    coords = {'time': [datetime(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)


# AWT: PCs (annual data, parse to daily)
# TODO: ESTE SALE DE LAS COPULAS
p_mat = op.join(p_data, 'AWT_PCs_500_part1.mat')
d_mat = rmat(p_mat)['AWT']
xds_PCs_sim = xr.Dataset(
    {
        'PC1': (('time',), d_mat['PCs'][:,0]),
        'PC2': (('time',), d_mat['PCs'][:,1]),
        'PC3': (('time',), d_mat['PCs'][:,2]),
    },
    coords = {'time': [datetime(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)
# reindex annual data to daily data
xds_PCs_sim = xr_daily(xds_PCs_sim)



# --------------------------------------
# Mount covariates matrix

# available data:
# model fit: xds_KMA_fit, xds_MJO_fit, xds_PCs_fit
# model sim: xds_MJO_sim, xds_PCs_sim

# covariates: FIT
d_covars_fit = xcd_daily(xds_MJO_fit, xds_PCs_fit)

# PCs covar 
cov_PCs = xds_PCs_fit.sel(time=slice(d_covars_fit[0],d_covars_fit[-1]))
cov_1 = cov_PCs.PC1.values.reshape(-1,1)
cov_2 = cov_PCs.PC2.values.reshape(-1,1)
cov_3 = cov_PCs.PC3.values.reshape(-1,1)

# MJO covars
cov_MJO = xds_MJO_fit.sel(time=slice(d_covars_fit[0],d_covars_fit[-1]))
cov_4 = cov_MJO.rmm1.values.reshape(-1,1)
cov_5 = cov_MJO.rmm2.values.reshape(-1,1)

# join covars and norm.
cov_T = np.hstack((cov_1, cov_2, cov_3, cov_4, cov_5))

# KMA related covars starting at KMA period 
i0 = d_covars_fit.index(x2d(xds_KMA_fit.time[0]))
cov_KMA = cov_T[i0:,:]
d_covars_fit = d_covars_fit[i0:]

# normalize
cov_norm_fit = (cov_KMA - cov_T.mean(axis=0)) / cov_T.std(axis=0)
xds_cov_fit = xr.Dataset(
    {
        'cov_norm': (('time','n_covariates'), cov_norm_fit),
    },
    coords = {
        'time': d_covars_fit,
    }
)

# covariates: SIMULATION
d_covars_sim = xcd_daily(xds_MJO_sim, xds_PCs_sim)

# PCs covar 
cov_PCs = xds_PCs_sim.sel(time=slice(d_covars_sim[0],d_covars_sim[-1]))
cov_1 = cov_PCs.PC1.values.reshape(-1,1)
cov_2 = cov_PCs.PC2.values.reshape(-1,1)
cov_3 = cov_PCs.PC3.values.reshape(-1,1)

# MJO covars
cov_MJO = xds_MJO_sim.sel(time=slice(d_covars_sim[0],d_covars_sim[-1]))
cov_4 = cov_MJO.rmm1.values.reshape(-1,1)
cov_5 = cov_MJO.rmm2.values.reshape(-1,1)

# join covars (do not normalize simulation covariates)
cov_T_sim = np.hstack((cov_1, cov_2, cov_3, cov_4, cov_5))
xds_cov_sim = xr.Dataset(
    {
        'cov_T': (('time','n_covariates'), cov_T_sim),
    },
    coords = {
        'time': d_covars_sim,
    }
)



# --------------------------------------
# Autoregressive Logistic Regression

# available data:
# model fit: xds_KMA_fit, xds_cov_sim, num_clusters
# model sim: xds_cov_sim, sim_num, sim_years


# use bmus inside covariate time frame
d_covars_bmus_fit = [
        max(d_covars_fit[0], x2d(xds_KMA_fit.time[0])),
        min(d_covars_fit[-1], x2d(xds_KMA_fit.time[-1]))]

xds_bmus_fit = xds_KMA_fit.sel(
    time=slice(d_covars_bmus_fit[0], d_covars_bmus_fit[-1])
).bmus


# Autoregressive logistic wrapper
num_clusters = 42
ALRW = ALR_WRP(xds_bmus_fit, num_clusters)

# ALR terms
d_terms_settings = {
    'mk_order'  : 1,
    'constant' : True,
    'long_term' : False,
    'seasonality': (True, [2, 4]),
    'covariates': (True, xds_cov_fit.cov_norm.values),
}

ALRW.SetFittingTerms(d_terms_settings)


# name test 
name_test = 'mk_test'
fit_and_save = True # False for loading


# ALR model fitting
p_save = op.join(p_data, '{0}.sav'.format(name_test))
fit_and_save = False
if fit_and_save:
    ALRW.FitModel(max_iter=20000)
    ALRW.SaveModel(p_save)
else:
    ALRW.LoadModel(p_save)

# Plot model p-values and params
p_report = op.join(p_data, 'r_{0}'.format(name_test))
ALRW.Report_pvalue(p_report)

# ALR model simulations 
sim_num = 2
sim_years = 300

# start simulation at PCs available data
d1 = x2d(xds_cov_sim.time[0])
d2 = datetime(d1.year+sim_years, d1.month, d1.day)
dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]


# print some info
print 'ALR model fit   : {0} --- {1}'.format(
    d_covars_bmus_fit[0], d_covars_bmus_fit[-1])
print 'ALR model sim   : {0} --- {1}'.format(
    dates_sim[0], dates_sim[-1])


# launch simulation
xds_ALR = ALRW.Simulate(
    sim_num, dates_sim, cov_T_sim)


# Save results for matlab plot 
evbmus_sim = xds_ALR.evbmus_sim.values
evbmus_probcum = xds_ALR.evbmus_probcum.values

p_mat_output = op.join(
    p_data, '{0}_y{1}s{2}.h5'.format(
        name_test, sim_years, sim_num))
import h5py
with h5py.File(p_mat_output, 'w') as hf:
    hf['bmusim'] = evbmus_sim
    hf['probcum'] = evbmus_probcum
    hf['dates'] = np.vstack(
        ([d.year for d in dates_sim],
        [d.month for d in dates_sim],
        [d.day for d in dates_sim])).T

# TODO: INTRODUCIR AQUI UN PLOT_COMPARE_PCS ?

