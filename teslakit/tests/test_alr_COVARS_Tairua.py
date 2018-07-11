#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import xarray as xr
import numpy as np
from datetime import datetime

# tk libs
from lib.objs.alr_wrapper import ALR_WRP
from lib.io.matlab import ReadMatfile as rmat
from lib.custom_dateutils import xds_common_dates_daily as xcd_daily
from lib.custom_dateutils import xds_reindex_daily as xr_daily
from lib.custom_dateutils import xds2datetime as x2d


# data storage
p_data = op.join(op.dirname(__file__),'..','data')
p_tests_Tairua = op.join(p_data,'tests','tests_ALR','tests_Tairua')
p_tairua_KMA = op.join(p_tests_Tairua, 'data','DWT_NZ_16.mat')
p_covars = op.join(p_tests_Tairua, 'covars')
p_output = op.join(p_tests_Tairua, 'output')


# --------------------------------------
# Get data used to FIT and SIMULATE

# KMA: bmus
d_mat = rmat(p_tairua_KMA)['KMA']
xds_KMA = xr.Dataset(
    {
        'bmus':(('time',), d_mat['bmus']),
    },
    coords = {'time': [datetime(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)


# MJO: rmm1, rmm2 (first date 1979-01-01 in order to avoid nans)
p_mat = op.join(p_covars, 'MJO.mat')
d_mat = rmat(p_mat)
xds_MJO = xr.Dataset(
    {
        'rmm1': (('time',), d_mat['rmm1']),
        'rmm2': (('time',), d_mat['rmm2']),
    },
    coords = {'time': [datetime(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)
# reindex to daily data after 1979-01-01 (avoid NaN) 
xds_MJO = xr_daily(xds_MJO, datetime(1979,01,01))


# AWT: PCs (annual data, parse to daily)
p_mat = op.join(p_covars, 'PCs_for_AWT.mat')
d_mat = rmat(p_mat)['AWT']
xds_PCs = xr.Dataset(
    {
        'PC1': (('time',), d_mat['PCs'][:,0]),
        'PC2': (('time',), d_mat['PCs'][:,1]),
        'PC3': (('time',), d_mat['PCs'][:,2]),
    },
    coords = {'time': [datetime(r[0],r[1],r[2]) for r in d_mat['Dates']]}
)
# reindex annual data to daily data
xds_PCs = xr_daily(xds_PCs)


# --------------------------------------
# Separate FITTING and SIMULATION covariates

cov_names = ['PC1', 'PC2', 'PC3', 'MJO1', 'MJO2']

# FITTING (1993-2012, KMA.bmus time)
d_covars_fit = xcd_daily([xds_MJO, xds_PCs, xds_KMA])

# PCs covar 
cov_PCs = xds_PCs.sel(time=slice(d_covars_fit[0],d_covars_fit[-1]))
cov_1 = cov_PCs.PC1.values.reshape(-1,1)
cov_2 = cov_PCs.PC2.values.reshape(-1,1)
cov_3 = cov_PCs.PC3.values.reshape(-1,1)

# MJO covars
cov_MJO = xds_MJO.sel(time=slice(d_covars_fit[0],d_covars_fit[-1]))
cov_4 = cov_MJO.rmm1.values.reshape(-1,1)
cov_5 = cov_MJO.rmm2.values.reshape(-1,1)

# join covars and norm.
cov_T = np.hstack((cov_1, cov_2, cov_3, cov_4, cov_5))

# Generate fitting covariates xr.Dataset
xds_cov_fit = xr.Dataset(
    {
        'cov_values': (('time','cov_names'), cov_T),
    },
    coords = {
        'time': d_covars_fit,
        'cov_names': cov_names,
    }
)

# SIMULATION (1979-2016)
d_covars_sim = xcd_daily([xds_MJO, xds_PCs])

# PCs covar 
cov_PCs = xds_PCs.sel(time=slice(d_covars_sim[0],d_covars_sim[-1]))
cov_1 = cov_PCs.PC1.values.reshape(-1,1)
cov_2 = cov_PCs.PC2.values.reshape(-1,1)
cov_3 = cov_PCs.PC3.values.reshape(-1,1)

# MJO covars
cov_MJO = xds_MJO.sel(time=slice(d_covars_sim[0],d_covars_sim[-1]))
cov_4 = cov_MJO.rmm1.values.reshape(-1,1)
cov_5 = cov_MJO.rmm2.values.reshape(-1,1)

# join covars 
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

# use bmus inside covariate time frame
xds_bmus_fit = xds_KMA.sel(
    time=slice(d_covars_fit[0], d_covars_fit[-1])
)

# covariates
xds_cov_fit = xds_cov_fit.sel(
    time=slice(d_covars_fit[0], d_covars_fit[-1])
)


# TEST parameters
name_test = 'test3'
num_clusters = 16
fit_and_save = True
sim_and_save = True
num_sims = 4

# ALR terms
#cov_season = [True, True, True, False, False]
d_terms_settings = {
    #'mk_order'  : 1,
    'constant' : True,
    #'long_term' : False,
    'seasonality': (True, [2, 4]),
    'covariates': (True, xds_cov_fit),
    #'covariates_seasonality': (True, cov_season),
}


# Autoregressive logistic wrapper
ALRW = ALR_WRP(xds_bmus_fit, num_clusters)
ALRW.SetFittingTerms(d_terms_settings)

# ALR model fitting
p_output_folder = op.join(p_output, name_test)
p_save = op.join(p_output_folder, '{0}.sav'.format(name_test))
if fit_and_save:
    if not op.isdir(p_output_folder):
        os.makedirs(p_output_folder)
    ALRW.FitModel(max_iter=20000)
    ALRW.SaveModel(p_save)
else:
    ALRW.LoadModel(p_save)

# Plot model p-values and params
p_report = op.join(p_output_folder, 'report_fitting')
ALRW.Report_Fit(p_report)


# ALR model simulations 
p_xds_output = op.join(
    p_output_folder, '{0}_s{1}.nc'.format(
        name_test, num_sims))
if sim_and_save:
    xds_ALR = ALRW.Simulate(num_sims, d_covars_sim, xds_cov_sim)

    # save results to netcdf
    xds_ALR.to_netcdf(p_xds_output, 'w')
else:
    xds_ALR = xr.open_dataset(p_xds_output)

# report SIM
p_report = op.join(p_output_folder, 'report_sim')
ALRW.Report_Sim(xds_ALR, p_report, xds_cov_sim)

