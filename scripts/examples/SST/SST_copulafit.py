#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op

# pip
import xarray as xr
from datetime import datetime, timedelta
import numpy as np
from datetime import date, timedelta, datetime

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../../')

# teslakit
from teslakit.project_site import Site
from teslakit.statistical import ksdensity_CDF, ksdensity_ICDF, copulafit, copularnd
from teslakit.alr import ALR_WRP
from teslakit.util.time_operations import xds_reindex_daily as xr_daily
from teslakit.io.aux_nc import StoreBugXdset as sbxds


# TODO: revisar

# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')
site.Summary()

# input files
p_sst_KMA = site.pc.site.sst.KMA

# output files
p_sst_alrw = site.pc.site.sst.alrw
p_PCs_sim = site.pc.site.sst.PCs_sim

# parameters
num_clusters = int(site.params.SST_AWT.num_clusters)

# Simulation dates (ALR)
d1_sim = np.datetime64(site.params.SIMULATION.date_ini).astype(datetime)
d2_sim = np.datetime64(site.params.SIMULATION.date_end).astype(datetime)
y1_sim = d1_sim.year
y2_sim = d2_sim.year


# --------------------------------------
# CALCULATE PC_SIM 1,2,3

# Load data
xds_AWT = xr.open_dataset(p_sst_KMA)

# bmus and order
kma_order = xds_AWT.order.values
kma_labels = xds_AWT.bmus.values

# first 3 PCs
PCs = xds_AWT.PCs.values
variance = xds_AWT.variance.values
PC1 = np.divide(PCs[:,0], np.sqrt(variance[0]))
PC2 = np.divide(PCs[:,1], np.sqrt(variance[1]))
PC3 = np.divide(PCs[:,2], np.sqrt(variance[2]))

# for each WT: generate copulas and simulate data 
d_pcs_wt = {}
for i in range(num_clusters):

    # getting copula number from plotting order
    num = kma_order[i]

    # find all the best match units equal
    ind = np.where(kma_labels == num)[:]

    # transfom data using kernel estimator
    cdf_PC1 = ksdensity_CDF(PC1[ind])
    cdf_PC2 = ksdensity_CDF(PC2[ind])
    cdf_PC3 = ksdensity_CDF(PC3[ind])
    U = np.column_stack((cdf_PC1.T, cdf_PC2.T, cdf_PC3.T))

    # fit PCs CDFs to a gaussian copula 
    # TODO: programar t-student
    rhohat, _ = copulafit(U, 'gaussian')

    # simulate data to fill probabilistic space
    U_sim = copularnd('gaussian', rhohat, 1000)
    PC1_rnd = ksdensity_ICDF(PC1[ind], U_sim[:,0])
    PC2_rnd = ksdensity_ICDF(PC2[ind], U_sim[:,1])
    PC3_rnd = ksdensity_ICDF(PC3[ind], U_sim[:,2])

    # store it
    # TODO: num o i ?
    d_pcs_wt['wt_{0}'.format(num+1)] = np.column_stack((PC1_rnd, PC2_rnd, PC2_rnd))


# --------------------------------------
# Autoregressive Logistic Regression - fit model
num_wts = 6
xds_bmus_fit = xr.Dataset(
    {
        'bmus':(('time',), xds_AWT.bmus),
    },
    coords = {'time': xds_AWT.time.values}
)

# ALR terms
d_terms_settings = {
    'mk_order'  : 1,
    'constant' : True,
    'long_term' : False,
    'seasonality': (False, []),
}

# ALR wrapper
ALRW = ALR_WRP(p_sst_alrw)
ALRW.SetFitData(num_wts, xds_bmus_fit, d_terms_settings)

# ALR model fitting
ALRW.FitModel(max_iter=10000)


# --------------------------------------
# Autoregressive Logistic Regression - simulate 

# simulation dates (year array)
dates_sim = [datetime(y,01,01) for y in range(y1_sim,y2_sim+1)]

# launch simulation
sim_num = 1
xds_alr = ALRW.Simulate(sim_num, dates_sim)
evbmus_sim = np.squeeze(xds_alr.evbmus_sims.values[:])

# Generate random PCs
print('\nGenerating PCs simulation: PC1, PC2, PC3 (random value withing category)...')
pcs123_sim = np.empty((len(evbmus_sim),3)) * np.nan
for c, m in enumerate(evbmus_sim):
    options = d_pcs_wt['wt_{0}'.format(int(m))]
    r = np.random.randint(options.shape[0])
    pcs123_sim[c,:] = options[r,:]

# store simulated PCs
xds_PCs_sim = xr.Dataset(
    {
        'PC1_rnd'  :(('time',), pcs123_sim[:,0]),
        'PC2_rnd'  :(('time',), pcs123_sim[:,1]),
        'PC3_rnd'  :(('time',), pcs123_sim[:,2]),
    },
    {'time' : dates_sim}
)

# Parse annual data to daily data
xds_PCs_sim = xr_daily(xds_PCs_sim)

# xarray.Dataset.to_netcdf() wont work with this time array and time dtype
sbxds(xds_PCs_sim, p_PCs_sim)
print('\nSST PCs Simulation stored at:\n{0}'.format(p_PCs_sim))



