#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op

# pip 
import xarray as xr
import numpy as np
from datetime import datetime, timedelta

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../../')

# teslakit 
from teslakit.project_site import PathControl
from teslakit.alr import ALR_WRP
from teslakit.io.matlab import ReadMatfile as rmat
from teslakit.util.time_operations import xds2datetime as x2d
from teslakit.util.time_operations import xds_reindex_daily as xr_daily
from teslakit.util.time_operations import xds_common_dates_daily as xcd_daily


# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ALR')

p_data = op.join(p_test, 'test_ALR_statsmodel')


# --------------------------------------
# Get data used to FIT ALR model and preprocess

# KMA: bmus (ESTELA + TCs)
p_mat = op.join(p_data, 'KMA_daily_42.mat')
d_mat = rmat(p_mat)['KMA']
xds_KMA_fit = xr.Dataset(
    {
        'bmus':(('time',), d_mat['bmus']),
    },
    coords = {'time': [datetime(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)


# MJO historical: rmm1, rmm2 (first date 1979-01-01 in order to avoid nans)
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
xds_MJO_fit = xr_daily(xds_MJO_fit, datetime(1979, 1, 1))


# AWT: PCs (SST annual data, parse to daily)
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

# MJO Simulated: rmm1, rmm2 (daily data)
p_mat = op.join(p_data, 'MJO_500_part1.mat')
d_mat = rmat(p_mat)
xds_MJO_sim = xr.Dataset(
    {
        'rmm1': (('time',), d_mat['rmm1']),
        'rmm2': (('time',), d_mat['rmm2']),
    },
    coords = {'time': [datetime(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)


# AWT: PCs (Generated with copula simulation. Annual data, parse to daily)
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
d_covars_fit = xcd_daily([xds_MJO_fit, xds_PCs_fit, xds_KMA_fit])

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

# generate xarray.Dataset
cov_names = ['PC1', 'PC2', 'PC3', 'MJO1', 'MJO2']
xds_cov_fit = xr.Dataset(
    {
        'cov_values': (('time','cov_names'), cov_T),
    },
    coords = {
        'time': d_covars_fit,
        'cov_names': cov_names,
    }
)


# covariates: SIMULATION
d_covars_sim = xcd_daily([xds_MJO_sim, xds_PCs_sim])

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
        'cov_values': (('time','cov_names'), cov_T_sim),
    },
    coords = {
        'time': d_covars_sim,
        'cov_names': cov_names,
    }
)



# --------------------------------------
# Autoregressive Logistic Regression

# available data:
# model fit: xds_KMA_fit, xds_cov_sim, num_clusters
# model sim: xds_cov_sim, sim_num, sim_years

# use bmus inside covariate time frame
xds_bmus_fit = xds_KMA_fit.sel(
    time=slice(d_covars_fit[0], d_covars_fit[-1])
)


# Autoregressive logistic wrapper
name_test = 'mk_test'
num_clusters = 42
sim_num = 2
fit_and_save = True # False for loading
p_test_ALR = op.join(p_data, name_test)

# ALR terms
d_terms_settings = {
    'mk_order'  : 1,
    'constant' : True,
    'long_term' : False,
    'seasonality': (True, [2, 4]),
    'covariates': (True, xds_cov_fit),
}


# Autoregressive logistic wrapper
ALRW = ALR_WRP(p_test_ALR)
ALRW.SetFitData(
    num_clusters, xds_bmus_fit, d_terms_settings)


# ALR model fitting
p_save = op.join(p_data, '{0}.sav'.format(name_test))
if fit_and_save:
    ALRW.FitModel(max_iter=20000)
else:
    ALRW.LoadModel(p_save)

# Plot model p-values and params
p_report = op.join(p_data, 'r_{0}'.format(name_test))
ALRW.Report_Fit(p_report)

# ALR model simulations 
sim_years = 300

# start simulation at PCs available data
d1 = x2d(xds_cov_sim.time[0])
d2 = datetime(d1.year+sim_years, d1.month, d1.day)
dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]


# print some info
print('ALR model fit   : {0} --- {1}'.format(
    d_covars_fit[0], d_covars_fit[-1]))
print('ALR model sim   : {0} --- {1}'.format(
    dates_sim[0], dates_sim[-1]))


# launch simulation
xds_ALR = ALRW.Simulate(
    sim_num, dates_sim, xds_cov_sim)


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


