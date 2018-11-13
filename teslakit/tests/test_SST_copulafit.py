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
from lib.statistical import Persistences, ksdensity_CDF
from lib.plotting.EOFs import Plot_EOFs_latavg as PlotEOFs
from lib.PCA import CalcPCA_latavg as CalcPCA
from lib.PCA import CalcRunningMean
from lib.objs.alr_wrapper import ALR_WRP


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')
site.Summary()

# input files
p_sst_PCA = site.pc.site.sst.PCA
p_sst_KMA = site.pc.site.sst.KMA

# parameters
num_clusters = int(site.params.SST_AWT.num_clusters)

# --------------------------------------
# TODO: CALCULATE PC_SIM 1,2,3

# Load data
xds_AWT = xr.open_dataset(p_sst_KMA)
print xds_AWT

# bmus and order
kma_order = xds_AWT.order.values
kma_labels = xds_AWT.bmus.values

# first 3 PCs
PCs = xds_AWT.PCs.values
variance = xds_AWT.variance.values
PC1 = np.divide(PCs[:,0], np.sqrt(variance[0]))
PC2 = np.divide(PCs[:,1], np.sqrt(variance[1]))
PC3 = np.divide(PCs[:,2], np.sqrt(variance[2]))

# TODO: PREGUNTAR ANA: entonces PC_rnd no depende de ALR output
# TODO generate copula for each WT
for i in range(num_clusters):

    # getting copula number from plotting order
    num = kma_order[i]

    # find all the best match units equal
    ind = np.where(kma_labels == num)[:]

    # transfom data using kernel estimator
    print PC1[ind]
    cdf_PC1 = ksdensity_CDF(PC1[ind])
    cdf_PC2 = ksdensity_CDF(PC2[ind])
    cdf_PC3 = ksdensity_CDF(PC3[ind])
    U = np.column_stack((cdf_PC1.T, cdf_PC2.T, cdf_PC3.T))

    # TODO: QUE HACEMOS?


# TODO: SE USA LA SIMULACION  PERO COMO?
sys.exit()

# Autoregressive Logistic Regression
xds_bmus_fit = xr.Dataset(
    {
        'bmus':(('time',), xds_AWT.bmus),
    },
    coords = {'time': xds_AWT.time.values}
).bmus

num_wts = 10
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

dates_sim = [
    datetime(x,1,1) for x in range(year_sim1,year_sim2+1)]

xds_ALR = ALRW.Simulate(sim_num, dates_sim)

# TODO: GUARDAR RESULTADOS
print xds_ALR
