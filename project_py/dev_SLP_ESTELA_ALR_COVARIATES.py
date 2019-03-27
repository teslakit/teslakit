#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
import ast
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import numpy as np
import xarray as xr
from datetime import date, timedelta, datetime

# tk libs
from lib.objs.tkpaths import Site
from lib.objs.alr_wrapper import ALR_WRP
from lib.custom_dateutils import xds_reindex_daily as xr_daily
from lib.custom_dateutils import xds_common_dates_daily as xcd_daily
from lib.custom_dateutils import xds2datetime as x2d
from lib.io.aux_nc import StoreBugXdset as sbxds


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')

DB = site.pc.DB                            # common database
ST = site.pc.site                          # site database
PR = site.params                           # site parameters

# input files
p_est_pred = ST.ESTELA.pred_slp
p_est_kma = op.join(p_est_pred, 'kma.nc')  # ESTELA + TCs Predictor
p_sst_PCA = ST.SST.pca                     # SST PCA
p_mjo_hist = DB.MJO.hist                   # historical MJO

p_sst_PCs_sim_d = ST.SST.pcs_sim_d         # daily PCs sim
p_mjo_sim =  ST.MJO.sim                    # daily MJO sim

# output files
p_alr_covars =  ST.ESTELA.alrw             # alr wrapper
p_dwt_sim = ST.ESTELA.sim_dwt              # daily weather types simulated with ALR

# ALR parameters
alr_markov_order = int(PR.SIMULATION.alr_covars_markov)
alr_seasonality = ast.literal_eval(PR.SIMULATION.alr_covars_seasonality)
n_sim = int(PR.SIMULATION.n_sim)


# --------------------------------------
# Get data used to FIT ALR model and preprocess

# KMA: bmus (daily)
xds_KMA_fit = xr.open_dataset(p_estela_kma)

# MJO: rmm1, rmm2 (daily)
xds_MJO_fit = xr.open_dataset(p_mjo_hist)


# SST: PCs (annual)
xds_PCs = xr.open_dataset(p_sst_PCA)
sst_PCs = xds_PCs.PCs.values[:]
xds_PCs_fit = xr.Dataset(
    {
        'PC1': (('time',), sst_PCs[:,0]),
        'PC2': (('time',), sst_PCs[:,1]),
        'PC3': (('time',), sst_PCs[:,2]),
    },
    coords = {'time': xds_PCs.time.values[:]}
)
# reindex annual data to daily data
xds_PCs_fit = xr_daily(xds_PCs_fit)



# --------------------------------------
# Get data used to SIMULATE ALR model and preprocess

# MJO: rmm1, rmm2 (daily data)
xds_MJO_sim = xr.open_dataset(p_mjo_sim)

# SST: PCs (daily data)
xds_PCs_sim = xr.open_dataset(p_sst_PCs_sim_d)



# --------------------------------------
# Mount covariates matrix

# available data:
# model fit: xds_KMA_fit, xds_MJO_fit, xds_PCs_fit
# model sim: xds_MJO_sim, xds_PCs_sim

# bmus fit
xds_BMUS_fit = xr.Dataset(
    {
        'bmus':(('time',), xds_KMA_fit['bmus'].values[:]),
    },
    coords = {'time': xds_KMA_fit.time.values[:]}
)

# covariates_fit
d_covars_fit = xcd_daily([xds_MJO_fit, xds_PCs_fit, xds_BMUS_fit])

# KMA dates
xds_BMUS_fit = xds_BMUS_fit.sel(time=slice(d_covars_fit[0],d_covars_fit[-1]))

# PCs covars 
cov_PCs = xds_PCs_fit.sel(time=slice(d_covars_fit[0],d_covars_fit[-1]))
cov_1 = cov_PCs.PC1.values.reshape(-1,1)
cov_2 = cov_PCs.PC2.values.reshape(-1,1)
cov_3 = cov_PCs.PC3.values.reshape(-1,1)

# MJO covars
cov_MJO = xds_MJO_fit.sel(time=slice(d_covars_fit[0],d_covars_fit[-1]))
cov_4 = cov_MJO.rmm1.values.reshape(-1,1)
cov_5 = cov_MJO.rmm2.values.reshape(-1,1)

# join covars 
# TODO: FER. NORMALIZO CONTRA TODO EL TIEMPO O EL TIEMPO DENTRO DE SIMULACION
cov_T = np.hstack((cov_1, cov_2, cov_3, cov_4, cov_5))

# normalize
cov_norm_fit = (cov_T - cov_T.mean(axis=0)) / cov_T.std(axis=0)
xds_cov_fit = xr.Dataset(
    {
        'cov_norm': (('time','n_covariates'), cov_norm_fit),
        'cov_names': (('n_covariates',), ['PC1','PC2','PC3','MJO1','MJO2']),
    },
    coords = {
        'time': d_covars_fit,
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
        'cov_values': (('time','n_covariates'), cov_T_sim),
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


# ALR terms
num_clusters = 42  # TODO NUM CLUSTERS AQUI EN .INI?
d_terms_settings = {
    'mk_order'  : alr_markov_order,
    'constant' : True,
    'long_term' : False,
    'seasonality': (True, alr_seasonality),
    'covariates': (True, xds_cov_fit),
}

# ALR wrapper
ALRW = ALR_WRP(p_alr_covars)
ALRW.SetFitData(num_clusters, xds_KMA_fit, d_terms_settings)

# ALR model fitting
fit_and_save = True
if fit_and_save:
    ALRW.FitModel(max_iter=20000)
else:
    ALRW.LoadModel()

# Plot model p-values and params
#ALRW.Report_Fit()

# simulate at covars available data 
dates_sim = d_covars_sim

# launch simulation
xds_alr = ALRW.Simulate(sim_num, dates_sim, xds_cov_sim)

# Store Daily Weather Types
xds_DWT_sim = xds_alr.evbmus_sims.to_dataset()
print(xds_DWT_sim)

# xarray.Dataset.to_netcdf() wont work with this time array and time dtype
sbxds(xds_DWT_sim, p_dwt_sim)
print('\nDWTs Simulation stored at:\n{0}'.format(p_dwt_sim))


