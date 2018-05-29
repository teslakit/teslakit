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
from lib.KMA import KMA_simple
from lib.statistical import Persistences, ksdensity_CDF
from lib.custom_plot import Plot_EOFs_latavg as PlotEOFs
from lib.PCA import CalcPCA_latavg as CalcPCA
from lib.objs.alr_wrapper import ALR_WRP


# data storage
p_data = op.join(op.dirname(__file__),'..','data')
p_export_figs = op.join(op.dirname(__file__),'..','data','export_figs')
p_pred_nc = op.join(p_data, 'SST_1854_2017.nc')


# --------------------------------------
# load predictor
xds_pred = xr.open_dataset(p_pred_nc)

# --------------------------------------
# Principal Components Analysis
pred_name = 'SST'
y1 = 1880
yN = 2016
m1 = 6
mN = 5

xds_PCA = CalcPCA(xds_pred, pred_name, y1, yN, m1, mN)
# TODO: ESTE OUTPUT SE USA EN TEST_ALR_COVARS

# plot EOFs
n_plot = 3
p_export = op.join(p_export_figs, 'latavg_EOFs')  # if only show: None
PlotEOFs(xds_PCA, n_plot, p_export)


# --------------------------------------
# KMA Classification 
num_clusters = 6
repres = 0.95

# TODO: SACAR COPULAS de DENTRO y poner aqui fuera
xds_AWT = KMA_simple(
    xds_PCA, num_clusters, repres)

# Get more data from xds_AWT

# add yearly time data to xds_AWT
time_yearly = [datetime(x,1,1) for x in range(y1,yN+1)]
xds_AWT['time']=(('n_pcacomp'), time_yearly)
# TODO: xds_AWT y xds_PCA SE USAN EN EL PROCESO E1A_MMSL_KWA.m

# Get bmus Persistences
# TODO: ver como guardar esta info / donde se usa?
d_pers_bmus = Persistences(xds_AWT.bmus.values)

# first 3 PCs
PCs = xds_AWT.PCs.values
variance = xds_AWT.variance.values
PC1 = np.divide(PCs[:,0], np.sqrt(variance[0]))
PC2 = np.divide(PCs[:,1], np.sqrt(variance[1]))
PC3 = np.divide(PCs[:,2], np.sqrt(variance[2]))

# TODO generate copula for each WT
for i in []:  #range(num_clusters):

    # getting copula number from plotting order
    num = kma_order[i]

    # find all the best match units equal
    ind = np.where(kma.labels_ == num)[:]

    # transfom data using kernel estimator
    cdf_PC1 = ksdensity_CDF(PC1[ind])
    cdf_PC2 = ksdensity_CDF(PC2[ind])
    cdf_PC3 = ksdensity_CDF(PC3[ind])
    U = np.column_stack((cdf_PC1.T, cdf_PC2.T, cdf_PC3.T))

    # TODO COPULAFIT. fit u to a student t copula. leer la web que compara

    # TODO COPULARND para crear USIMULADO

    # TODO: KS DENSITY ICDF PARA CREAR PC123_RND SIMULATODS

    # TODO: USAR NUM PARA GUARDAR LOS RESULTADOS




# --------------------------------------
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
year_sim1 = 1700
year_sim2 = 2700

dates_sim = [
    datetime(x,1,1) for x in range(year_sim1,year_sim2+1)]

xds_ALR = ALRW.Simulate(sim_num, dates_sim)

print xds_ALR

