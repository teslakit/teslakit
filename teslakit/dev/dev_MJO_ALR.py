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
from lib.mjo import GetMJOCategories
from lib.plotting.MJO import Plot_MJOphases, Plot_MJOCategories
from lib.objs.alr_wrapper import ALR_WRP
from lib.io.aux_nc import StoreBugXdset as sbxds


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')
site.Summary()

# input files
p_mjo_hist = site.pc.DB.mjo.hist

# output files
p_mjo_alrw = site.pc.site.mjo.alrw
p_mjo_sim =  site.pc.site.mjo.sim

# export figs
p_export_mjo = site.pc.site.exp.mjo

# MJO ALR parameters
alr_markov_order = int(site.params.MJO.alr_markov)
alr_seasonality = ast.literal_eval(site.params.MJO.alr_seasonality)

# Simulation dates (ALR)
d1_sim = np.datetime64(site.params.SIMULATION.date_ini).astype(datetime)
d2_sim = np.datetime64(site.params.SIMULATION.date_end).astype(datetime)


# --------------------------------------
# Load MJO data (previously downloaded)
xds_mjo_hist = xr.open_dataset(p_mjo_hist)


# --------------------------------------
# Calculate MJO categories (25 used) 
print('\nCalculating MJO categories (from 25 options)...')
rmm1 = xds_mjo_hist['rmm1']
rmm2 = xds_mjo_hist['rmm2']
phase = xds_mjo_hist['phase']

categ, d_rmm_categ = GetMJOCategories(rmm1, rmm2, phase)
xds_mjo_hist['categ'] = (('time',), categ)


# plot MJO phases
p_export = op.join(p_export_mjo, 'mjo_phases')  # if only show: None
Plot_MJOphases(rmm1, rmm2, phase, p_export)

# plot MJO categories
p_export = op.join(p_export_mjo, 'mjo_categ')  # if only show: None
Plot_MJOCategories(rmm1, rmm2, categ, p_export)


# --------------------------------------
# Autoregressive Logistic Regression - fit model

print(
'\nSetting ALR execution parameters...\n \
markov_order = {0}\n seasonality = {1}'.format(
    alr_markov_order, alr_seasonality)
)

# MJO historical data for fitting
num_categs  = 25  # fixed parameter

xds_bmus_fit = xr.Dataset(
    {
        'bmus'  :(('time',), xds_mjo_hist.categ.values[:]),
    },
    {'time' : xds_mjo_hist.time}
)

# ALR terms
d_terms_settings = {
    'mk_order'  : alr_markov_order,
    'constant' : True,
    'seasonality': (True, alr_seasonality),
}

# ALR wrapper
ALRW = ALR_WRP(p_mjo_alrw)
ALRW.SetFitData(num_categs, xds_bmus_fit, d_terms_settings)

# ALR model fitting
ALRW.FitModel(max_iter=10000)
ALRW.Report_Fit()


# --------------------------------------
# Autoregressive Logistic Regression - simulate 

# simulation dates
dates_sim = [d1_sim + timedelta(days=i) for i in range((d2_sim-d1_sim).days+1)]

# launch simulation
sim_num = 1  # only one simulation for mjo daily
xds_alr = ALRW.Simulate(sim_num, dates_sim)
evbmus_sim = np.squeeze(xds_alr.evbmus_sims.values[:])


# Generate mjo rmm1 and rmm2 simulated values
print('\nGenerating MJO simulation: rmm1, rmm2 (random value withing category)...')
rmm12_sim = np.empty((len(evbmus_sim),2)) * np.nan
for c, m in enumerate(evbmus_sim):
    options = d_rmm_categ['cat_{0}'.format(int(m))]
    r = np.random.randint(options.shape[0])
    rmm12_sim[c,:] = options[r,:]

# mjo and phase
mjo_sim = np.sqrt(rmm12_sim[:,0]**2+rmm12_sim[:,1]**2)
phase_sim = np.arctan2(rmm12_sim[:,0], rmm12_sim[:,1])

# store simulated mjo
xds_MJO_sim = xr.Dataset(
    {
        'mjo'   :(('time',), mjo_sim),
        'phase' :(('time',), phase_sim),
        'rmm1'  :(('time',), rmm12_sim[:,0]),
        'rmm2'  :(('time',), rmm12_sim[:,1]),
    },
    {'time' : dates_sim}
)

print xds_MJO_sim

# xarray.Dataset.to_netcdf() wont work with this time array and time dtype
sbxds(xds_MJO_sim, p_mjo_sim)
print('\nMJO Simulation stored at:\n{0}'.format(p_mjo_sim))

