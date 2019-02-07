#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import pickle
import xarray as xr
from datetime import datetime, timedelta
import numpy as np

# tk libs
from lib.objs.tkpaths import Site
from lib.KMA import KMA_simple
from lib.statistical import Persistences, CopulaSimulation 
from lib.plotting.EOFs import Plot_EOFs_latavg as PlotEOFs
from lib.plotting.KMA import Plot_Weather_Types, Plot_WTs_Dates
from lib.plotting.KMA import Plot_3D_3PCs_WTs, Plot_Compare_WTs_hist
from lib.PCA import CalcPCA_latavg as CalcPCA
from lib.PCA import CalcRunningMean
from lib.objs.alr_wrapper import ALR_WRP
from lib.io.aux_nc import StoreBugXdset as sbxds
from lib.custom_dateutils import xds_reindex_daily as xr_daily
from lib.custom_dateutils import xds_reindex_monthly as xr_monthly


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')

DB = site.pc.DB                        # common database
ST = site.pc.site                      # site database
PR = site.params                       # site parameters

# input files
p_SST = DB.SST.hist_pacific            # SST Pacific area

# output files
p_export_figs = ST.export_figs.sst
p_sst_PCA = ST.SST.pca
p_sst_KMA = ST.SST.kma
p_sst_alrw = ST.SST.alrw
p_PCs_sim = ST.SST.pcs_sim
p_PCs_sim_d = ST.SST.pcs_sim_d
p_PCs_sim_m = ST.SST.pcs_sim_m

# PCA dates parameters
pred_name = 'SST'
y1 = int(PR.SST_AWT.pca_year_ini)
yN = int(PR.SST_AWT.pca_year_end)
m1 = int(PR.SST_AWT.pca_month_ini)
mN = int(PR.SST_AWT.pca_month_end)
num_clusters = int(PR.SST_AWT.num_clusters)
repres = float(PR.SST_AWT.repres)
num_PCs_rnd = int(PR.SST_AWT.num_pcs_rnd)

# Simulation dates (ALR)
y1_sim = int(PR.SIMULATION.date_ini.split('-')[0])
y2_sim = int(PR.SIMULATION.date_end.split('-')[0])


# --------------------------------------
# load SST predictor from database
xds_pred = xr.open_dataset(p_SST)

# Calculate running average
print('\nCalculating {0} running average... '.format(pred_name))
xds_pred = CalcRunningMean(xds_pred, pred_name)

# Principal Components Analysis
print('\nPrincipal Component Analysis (latitude average)...')
xds_PCA = CalcPCA(xds_pred, pred_name, y1, yN, m1, mN)

# plot EOFs
n_plot = 6
p_export = op.join(p_export_figs, 'latavg_EOFs')  # if only show: None
PlotEOFs(xds_PCA, n_plot, p_export)


# --------------------------------------
# KMA Classification 
print('\nKMA Classification...')
xds_AWT = KMA_simple(
    xds_PCA, num_clusters, repres)
# TODO: resultado KMA a veces funciona a veces no????? 

# PCA, KMA  dates (annual array)
dates_fit = [datetime(y,m1,1) for y in range(y1,yN+1)]
xds_PCA['time']=(('n_components'), dates_fit)
xds_AWT['time']=(('n_pcacomp'), dates_fit)

# store AWTs and PCs
xds_PCA.to_netcdf(p_sst_PCA,'w')  # store SST PCA data 
xds_AWT.to_netcdf(p_sst_KMA,'w')  # store SST KMA data 
print('\n{0} PCA and KMA stored at:\n{1}\n{2}'.format(
    pred_name, p_sst_PCA, p_sst_KMA))

# Plot Weather Types
p_export = op.join(p_export_figs, 'AWT_WeatherTypes.png')
Plot_Weather_Types(xds_AWT, xds_PCA.pred_lon, p_export)

# Plot year/label wts
p_export = op.join(p_export_figs, 'AWT_WT_Dates.png')
Plot_WTs_Dates(xds_AWT, p_export)


# --------------------------------------
# Get more data from xds_AWT
kma_order = xds_AWT.order.values
kma_labels = xds_AWT.bmus.values

# Get bmus Persistences
# TODO: guardar con pickle. donde se usa?
# calcular para simulado?
#d_pers_bmus = Persistences(xds_AWT.bmus.values)

# first 3 PCs
PCs = xds_AWT.PCs.values
variance = xds_AWT.variance.values
PC1 = np.divide(PCs[:,0], np.sqrt(variance[0]))
PC2 = np.divide(PCs[:,1], np.sqrt(variance[1]))
PC3 = np.divide(PCs[:,2], np.sqrt(variance[2]))

# for each WT: generate copulas and simulate data 
d_pcs_fit = {}
d_pcs_rnd = {}
for i in range(num_clusters):

    # getting copula number from plotting order
    num = kma_order[i]

    # find all the best match units equal
    ind = np.where(kma_labels == num)[:]

    # PCs for weather type
    PC123 = np.column_stack((PC1[ind], PC2[ind], PC3[ind]))

    # statistical simulate PCs using copulas with KDE (kernel density estimation)
    kernels = ['KDE', 'KDE', 'KDE']
    PC123_rnd = CopulaSimulation(PC123, kernels, num_PCs_rnd)

    # store data 
    d_pcs_fit['WT #{0}'.format(num+1)] = PC123
    d_pcs_rnd['WT #{0}'.format(num+1)] = PC123_rnd


# store WTS PC123 fit and rnd_generation
p_pick = op.join(p_export_figs, 'd_pcs_fit.pickle')
pickle.dump(d_pcs_fit, open(p_pick, 'wb'))
p_pick = op.join(p_export_figs, 'd_pcs_rnd.pickle')
pickle.dump(d_pcs_rnd, open(p_pick, 'wb'))


# Plot Weather Type 3D PCs for fit and random generation data
p_export = op.join(p_export_figs, 'AWT_WTs_3DPCs_fit.png')
Plot_3D_3PCs_WTs(d_pcs_fit, 'Weather Types PCs (fit)', p_export)

p_export = op.join(p_export_figs, 'AWT_WTs_3DPCs_rnd.png')
Plot_3D_3PCs_WTs(d_pcs_rnd, 'Weather Types PCs (rnd)', p_export)

# Plot Weather Type histogram comparison for fit and random generation data
p_export = op.join(p_export_figs, 'AWT_WTs_Histograms')  # folder
Plot_Compare_WTs_hist(d_pcs_fit, d_pcs_rnd, p_export)


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

# show fit report
#ALRW.Report_Fit(export=True)


# --------------------------------------
# Autoregressive Logistic Regression - simulate 

# simulation dates (annual array)
dates_sim = [datetime(y,m1,1) for y in range(y1_sim,y2_sim+1)]

# launch simulation
sim_num = 1
xds_alr = ALRW.Simulate(sim_num, dates_sim)
evbmus_sim = np.squeeze(xds_alr.evbmus_sims.values[:])

# Generate random PCs
print('\nGenerating PCs simulation: PC1, PC2, PC3 (random value withing category)...')
pcs123_sim = np.empty((len(evbmus_sim),3)) * np.nan
for c, m in enumerate(evbmus_sim):
    options = d_pcs_rnd['WT #{0}'.format(int(m))]
    r = np.random.randint(options.shape[0])
    pcs123_sim[c,:] = options[r,:]

# denormalize simulated PCs
PC1_sim = np.multiply(pcs123_sim[:,0], np.sqrt(variance[0]))
PC2_sim = np.multiply(pcs123_sim[:,1], np.sqrt(variance[1]))
PC3_sim = np.multiply(pcs123_sim[:,2], np.sqrt(variance[2]))

# store simulated PCs
xds_PCs_sim = xr.Dataset(
    {
        'PC1'  :(('time',), PC1_sim),
        'PC2'  :(('time',), PC2_sim),
        'PC3'  :(('time',), PC3_sim),
    },
    {'time' : dates_sim}
)

# xarray.Dataset.to_netcdf() wont work with this time array and time dtype
sbxds(xds_PCs_sim, p_PCs_sim)
print('\nSST PCs Simulation (yearly) stored at:\n{0}'.format(p_PCs_sim))

# resample to daily and store
xds_PCs_sim_d = xr_daily(xds_PCs_sim)
sbxds(xds_PCs_sim_d, p_PCs_sim_d)
print('\nSST PCs Simulation (daily) stored at:\n{0}'.format(p_PCs_sim_d))

# resample to monthly and store
xds_PCs_sim_m = xr_monthly(xds_PCs_sim)
sbxds(xds_PCs_sim_m, p_PCs_sim_m)
print('\nSST PCs Simulation (monthly) stored at:\n{0}'.format(p_PCs_sim_m))

