#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# pandas - matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# python libs
import numpy as np
import xarray as xr
import pickle

# tk libs
from lib.objs.tkpaths import Site
from lib.objs.predictor import Predictor
from lib.io.matlab import ReadGowMat, ReadCoastMat, ReadEstelaMat
from lib.estela import spatial_gradient, mask_from_poly


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')

DB = site.pc.DB                           # common database
ST = site.pc.site                         # site database
PR = site.params                          # site parameters

# input files
p_est_coastmat = ST.ESTELA.coastmat       # estela coast (.mat)
p_est_datamat = ST.ESTELA.estelamat       # estela data (.mat)
p_gow_mat = ST.ESTELA.gowpoint            # gow point (.mat)
p_wvs_parts_p1 = ST.WAVES.partitions_p1
p_slp = ST.ESTELA.slp                     # site slp data (.nc)
p_hist_r2_params = ST.TCs.hist_r2_params  # hist storms inside r2 parameters

# output files
p_est_pred = ST.ESTELA.pred_slp           # estela slp predictor
p_dbins = ST.ESTELA.hydrographs           # intradaily mu tau hydrographs

# parameters for KMA_REGRESSION_GUIDED
kma_date_ini = site.params.ESTELA_KMA_RG.date_ini
kma_date_end = site.params.ESTELA_KMA_RG.date_end
num_clusters = int(site.params.ESTELA_KMA_RG.num_clusters)
kmarg_alpha = float(site.params.ESTELA_KMA_RG.alpha)


# --------------------------------------
# load sea polygons for mask generation
ls_sea_poly = ReadCoastMat(p_est_coastmat)

# load estela data 
xds_est = ReadEstelaMat(p_est_datamat)


# --------------------------------------
# load waves data from .mat gow file
xds_WAVES = ReadGowMat(p_gow_mat)
print('\nResampling waves data to daily mean...')

# calculate Fe (from GOW waves data)
hs = xds_WAVES.hs
tm = xds_WAVES.t02
Fe = np.multiply(hs**2,tm)**(1.0/3)
xds_WAVES.update({
    'Fe':(('time',), Fe)
})

# select time window and do data daily mean
xds_WAVES = xds_WAVES.sel(
    time = slice(kma_date_ini, kma_date_end)
).resample(time='1D').mean()
print('WVS: ',xds_WAVES.time.values[0],' - ',xds_WAVES.time.values[-1])


# --------------------------------------
# load SLP (use xarray saved predictor data) 
xds_SLP_site = xr.open_dataset(p_slp)

# select time window and do data daily mean
print('\nResampling SLP data to daily mean...')
xds_SLP_day = xds_SLP_site.sel(
    time = slice(kma_date_ini, kma_date_end)
).resample(time='1D').mean()
print('SLP: ',xds_SLP_day.time.values[0],' - ',xds_SLP_day.time.values[-1])

# calculate daily gradients
print('\nCalculating SLP spatial gradient...')
xds_SLP_day = spatial_gradient(xds_SLP_day, 'SLP')

# generate land mask with land polygons 
print('\nReading land mask and ESTELA data for SLP...')
xds_SLP_day = mask_from_poly(
    xds_SLP_day, ls_sea_poly, 'mask_land')

# resample estela to site mesh
xds_est_site = xds_est.sel(
    longitude = xds_SLP_site.longitude,
    latitude = xds_SLP_site.latitude,
)
estela_D = xds_est_site.D_y1993to2012

# generate SLP mask using estela
mask_est = np.zeros(estela_D.shape)
mask_est[np.where(estela_D<1000000000)]=1

xds_SLP_day.update({
    'mask_estela':(('latitude','longitude'), mask_est)
})


# --------------------------------------
# Use a tesla-kit predictor
pred = Predictor(p_est_pred)
pred.data = xds_SLP_day

# Calculate PCA (dynamic estela predictor)
print('\nPrincipal Component Analysis (ESTELA predictor)...')
pred.Calc_PCA_EstelaPred('SLP', estela_D)

# plot PCA EOFs
n_EOFs = 3
pred.Plot_EOFs_EstelaPred(n_EOFs, show=False)


# Calculate KMA (regression guided with WAVES data)
print('\nKMA Classification (regression guided: waves)...')
# TODO: encontrar alpha optimo?
pred.Calc_KMA_regressionguided(
    num_clusters,
    xds_WAVES, ['hs','t02','Fe'],
    kmarg_alpha)

# save predictor data
pred.Save()

# plot KMA clusters
pred.Plot_KMArg_clusters_datamean('SLP', show=False, mask_name='mask_estela')


# --------------------------------------
# load historical storms-parameters inside r2
xds_TCs_r2_params = xr.open_dataset(p_hist_r2_params)

storm_dates = xds_TCs_r2_params.dmin_date.values[:]
storm_categs = xds_TCs_r2_params.category.values[:]

# modify predictor KMA with circle storms data
print('\nAdding Historical TCs to SLP_PREDICTOR KMA_RG bmus...')
pred.Mod_KMA_AddStorms(storm_dates, storm_categs)


# --------------------------------------
# Calculate intradaily MU TAU hydrographs 
xds_WAVES_p1 = ReadGowMat(p_wvs_parts_p1)

print('\nCalculating MU TAU hydrographs...')
dict_bins = pred.Calc_MU_TAU_Hydrographs(xds_WAVES_p1)

# TODO: add hydrograph plots lib/plotting/intradaily

# store hydrographs
pickle.dump(dict_bins, open(p_dbins,'wb'))
print('\nMU, TAU hydrographs stored at:\n{0}'.format(p_dbins))

