#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import numpy as np
import xarray as xr

# tk libs
from lib.objs.tkpaths import Site
from lib.objs.predictor import Predictor
from lib.io.matlab import ReadGowMat, ReadCoastMat, ReadEstelaMat
from lib.estela import spatial_gradient, mask_from_poly
from lib.tcyclone import Extract_Circle


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')
site.Summary()

# input files
p_estela_coast_mat = site.pc.site.est.coastmat  # estela coast (.mat)
p_estela_data_mat = site.pc.site.est.estelamat  # estela data (.mat)
p_gow_mat = site.pc.site.est.gowpoint  # gow point (.mat)
p_slp = site.pc.site.est.slp  # site slp data (.nc)
p_hist_tcs = site.pc.DB.tcs.noaa_fix  # WMO historical TCs

# output files
p_estela_pred = site.pc.site.est.pred_slp  # estela slp predictor

# parameters for KMA_REGRESSION_GUIDED
waves_date_ini = site.params.ESTELA_KMA_RG.waves_date_ini
waves_date_end = site.params.ESTELA_KMA_RG.waves_date_end

# wave point lon, lat, and radius for TCs selection
pnt_lon = float(site.params.WAVES.point_longitude)
pnt_lat = float(site.params.WAVES.point_latitude)
r2 = float(site.params.TCS.r2)   # smaller one


# --------------------------------------
# load sea polygons for mask generation
ls_sea_poly = ReadCoastMat(p_estela_coast_mat)

# load estela data 
xds_est = ReadEstelaMat(p_estela_data_mat)


# --------------------------------------
# load waves data from .mat gow file
xds_WAVES = ReadGowMat(p_gow_mat)

# calculate Fe (from GOW waves data)
hs = xds_WAVES.hs
tm = xds_WAVES.t02
Fe = np.multiply(hs**2,tm)**(1.0/3)
xds_WAVES.update({
    'Fe':(('time',), Fe)
})

# select time window and do data daily mean
xds_WAVES = xds_WAVES.sel(
    time = slice(waves_date_ini, waves_date_end)
).resample(time='1D').mean()


# --------------------------------------
# load SLP (use xarray saved predictor data) 
xds_SLP_site = xr.open_dataset(p_slp)

# parse data to daily average 
xds_SLP_day = xds_SLP_site.resample(time='1D').mean()

# calculate daily gradients
xds_SLP_day = spatial_gradient(xds_SLP_day, 'SLP')

# generate land mask with land polygons 
xds_SLP_day = mask_from_poly(
    xds_SLP_day, ls_sea_poly, 'mask_land')

# resample estela to site mesh
xds_est_site = xds_est.sel(
    longitude = xds_SLP_day.longitude,
    latitude = xds_SLP_day.latitude,
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
pred = Predictor(p_estela_pred)
pred.data = xds_SLP_day


# Calculate PCA (dynamic estela predictor)
pred.Calc_PCA_EstelaPred('SLP', estela_D)

# plot PCA EOFs
n_EOFs = 3
pred.Plot_EOFs_EstelaPred(n_EOFs, show=False)


# Calculate KMA (regression guided with WAVES data)
num_clusters = 36
alpha = 0.3  # TODO: encontrar alpha optimo?
pred.Calc_KMA_regressionguided(
    num_clusters,
    xds_WAVES, ['hs','t02','Fe'],
    alpha)
pred.Save()

# plot KMA clusters
#pred.Plot_KMArg_clusters_datamean('SLP', show=True, mask_name='mask_estela')



# --------------------------------------
# load storms, find inside circle and modify predictor KMA 
xds_wmo_fix = xr.open_dataset(p_hist_tcs)

# extract TCs inside circle using GOW point as center 
print(
'\nExtracting Historical TCs from WMO database...\n \
Lon = {0:.2f}º , Lat = {1:.2f}º, R2  = {2:6.2f}º'.format(
    pnt_lon, pnt_lat, r2)
)

_, xds_in = Extract_Circle(
    xds_wmo_fix, pnt_lon, pnt_lat, r2)

storm_dates = xds_in.inside_date.values[:]
storm_categs = xds_in.category.values[:]

# modify predictor KMA with circle storms data
print('\nAdding Historical TCs to SLP_PREDICTOR KMA_RG bmus...')
pred.Mod_KMA_AddStorms(storm_dates, storm_categs)
pred.Save()

