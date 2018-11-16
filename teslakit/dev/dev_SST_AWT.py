#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import xarray as xr
from datetime import datetime, timedelta
import numpy as np

# tk libs
from lib.objs.tkpaths import Site
from lib.KMA import KMA_simple
from lib.statistical import Persistences, ksdensity_CDF, ksdensity_ICDF, copulafit, copularnd
from lib.plotting.EOFs import Plot_EOFs_latavg as PlotEOFs
from lib.PCA import CalcPCA_latavg as CalcPCA
from lib.PCA import CalcRunningMean
from lib.objs.alr_wrapper import ALR_WRP
from lib.io.aux_nc import StoreBugXdset as sbxds


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')
site.Summary()

# input files
p_SST = site.pc.DB.sst.hist_pacific  # SST Pacific area

# output files
p_export_figs = site.pc.site.exp.sst
p_sst_PCA = site.pc.site.sst.PCA
p_sst_KMA = site.pc.site.sst.KMA
p_sst_alrw = site.pc.site.sst.alrw
p_PCs_sim = site.pc.site.sst.PCs_sim

# PCA dates parameters
pred_name = 'SST'
y1 = int(site.params.SST_AWT.pca_year_ini)
yN = int(site.params.SST_AWT.pca_year_end)
m1 = int(site.params.SST_AWT.pca_month_ini)
mN = int(site.params.SST_AWT.pca_month_end)
num_clusters = int(site.params.SST_AWT.num_clusters)
repres = float(site.params.SST_AWT.repres)
num_PCs_rnd = int(site.params.SST_AWT.num_pcs_rnd)

# Simulation dates (ALR)
y1_sim = int(site.params.SIMULATION.date_ini.split('-')[0])
y2_sim = int(site.params.SIMULATION.date_end.split('-')[0])


# --------------------------------------
# load SST predictor from database
xds_pred = xr.open_dataset(p_SST)


# --------------------------------------
# Calculate running average
print('\nCalculating {0} running average... '.format(pred_name))
xds_pred = CalcRunningMean(xds_pred, pred_name)

# Principal Components Analysis
print('\nPrincipal Component Analysis (latitude average)...')
# TODO: RECORTAR AQUI LAS FECHAS / MONTAR AQUI EL VECTOR DE FECHAS
xds_PCA = CalcPCA(xds_pred, pred_name, y1, yN, m1, mN)

# plot EOFs
n_plot = 3
p_export = op.join(p_export_figs, 'latavg_EOFs')  # if only show: None
PlotEOFs(xds_PCA, n_plot, p_export)


# --------------------------------------
# KMA Classification 
print('\nKMA Classification...')
xds_AWT = KMA_simple(
    xds_PCA, num_clusters, repres)

# add yearly time data to xds_AWT and xds_PCA 
time_yearly = [datetime(x,1,1) for x in range(y1,yN+1)]
xds_PCA['time']=(('n_components'), time_yearly)
xds_AWT['time']=(('n_pcacomp'), time_yearly)

# store AWTs and PCs
xds_PCA.to_netcdf(p_sst_PCA,'w')  # store SST PCA data 
xds_AWT.to_netcdf(p_sst_KMA,'w')  # store SST KMA data 
print('\n{0} PCA and KMA stored at:\n{1}\n{2}'.format(
    pred_name, p_sst_PCA, p_sst_KMA))


# --------------------------------------
# Get more data from xds_AWT
kma_order = xds_AWT.order.values
kma_labels = xds_AWT.bmus.values


# Get bmus Persistences
# TODO: ver como guardar esta info / donde se usa?
d_pers_bmus = Persistences(xds_AWT.bmus.values)

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
    rhohat, _ = copulafit(U, 'gaussian')

    # simulate data to fill probabilistic space
    U_sim = copularnd('gaussian', rhohat, num_PCs_rnd)

    # get back PCs values from kernel estimator
    PC1_rnd = ksdensity_ICDF(PC1[ind], U_sim[:,0])
    PC2_rnd = ksdensity_ICDF(PC2[ind], U_sim[:,1])
    PC3_rnd = ksdensity_ICDF(PC3[ind], U_sim[:,2])

    # store data  # TODO : num o i????
    d_pcs_wt['wt_{0}'.format(num+1)] = np.column_stack((PC1_rnd, PC2_rnd, PC2_rnd))


# --------------------------------------
# Autoregressive Logistic Regression
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
ALRW.SetFitData(num_clusters, xds_bmus_fit, d_terms_settings)

# ALR model fitting
ALRW.FitModel(max_iter=10000)


# --------------------------------------
# Autoregressive Logistic Regression - simulate 

# simulation dates (annual array)
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
        'PC1'  :(('time',), pcs123_sim[:,0]),
        'PC2'  :(('time',), pcs123_sim[:,1]),
        'PC3'  :(('time',), pcs123_sim[:,2]),
    },
    {'time' : dates_sim}
)

# xarray.Dataset.to_netcdf() wont work with this time array and time dtype
sbxds(xds_PCs_sim, p_PCs_sim)
print('\nSST PCs Simulation stored at:\n{0}'.format(p_PCs_sim))

