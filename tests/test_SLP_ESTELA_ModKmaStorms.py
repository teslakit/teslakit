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
kma_date_ini = site.params.ESTELA_KMA_RG.date_ini
kma_date_end = site.params.ESTELA_KMA_RG.date_end
num_clusters = int(site.params.ESTELA_KMA_RG.num_clusters)
kmarg_alpha = float(site.params.ESTELA_KMA_RG.alpha)

# wave point lon, lat, and radius for TCs selection
pnt_lon = float(site.params.WAVES.point_longitude)
pnt_lat = float(site.params.WAVES.point_latitude)
r2 = float(site.params.TCS.r2)   # smaller one


# --------------------------------------
# Use a tesla-kit predictor
pred = Predictor(p_estela_pred)
pred.Load()


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

storm_dates = xds_in.dmin_date.values[:]
storm_categs = xds_in.category.values[:]

# modify predictor KMA with circle storms data
print('\nAdding Historical TCs to SLP_PREDICTOR KMA_RG bmus...')
pred.Mod_KMA_AddStorms(storm_dates, storm_categs)

