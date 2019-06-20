#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op
import sys

# pip
import numpy as np
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../../')

# teslakit
from teslakit.project_site import Site
from teslakit.estela import Predictor
from teslakit.io.matlab import ReadGowMat, ReadCoastMat, ReadEstelaMat
from teslakit.estela import spatial_gradient, mask_from_poly
from teslakit.storms import Extract_Circle


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')

DB = site.pc.DB                               # common database
ST = site.pc.site                             # site database
PR = site.params                              # site parameters

# input files
p_estela_coast_mat = ST.ESTELA.coastmat       # estela coast (.mat)
p_estela_data_mat = ST.ESTELA.estelamat       # estela data (.mat)
p_gow_mat = ST.ESTELA.gowpoint                # gow point (.mat)
p_slp = ST.ESTELA.slp                         # site slp data (.nc)
p_hist_tcs = DB.TCs.noaa                      # WMO historical TCs

# output files
p_estela_pred = ST.ESTELA.pred_slp            # estela slp predictor

# parameters for KMA_REGRESSION_GUIDED
kma_date_ini = PR.ESTELA_KMA_RG.date_ini
kma_date_end = PR.ESTELA_KMA_RG.date_end
num_clusters = int(PR.ESTELA_KMA_RG.num_clusters)
kmarg_alpha = float(PR.ESTELA_KMA_RG.alpha)

# wave point lon, lat, and radius for TCs selection
pnt_lon = float(PR.WAVES.point_longitude)
pnt_lat = float(PR.WAVES.point_latitude)
r2 = float(site.params.TCS.r2)   # smaller one


# --------------------------------------
# Use a tesla-kit predictor
pred = Predictor(p_estela_pred)
pred.Load()


# --------------------------------------
# load storms, find inside circle and modify predictor KMA 
xds_wmo = xr.open_dataset(p_hist_tcs)

# dictionary with needed variable names
d_vns = {
    'longitude': 'lon_wmo',
    'latitude': 'lat_wmo',
    'time': 'time_wmo',
    'pressure': 'pres_wmo',
}


# extract TCs inside circle using GOW point as center 
print(
'\nExtracting Historical TCs from WMO database...\n \
Lon = {0:.2f}º , Lat = {1:.2f}º, R2  = {2:6.2f}º'.format(
    pnt_lon, pnt_lat, r2)
)

_, xds_in = Extract_Circle(
    xds_wmo, pnt_lon, pnt_lat, r2, d_vns)

storm_dates = xds_in.dmin_date.values[:]
storm_categs = xds_in.category.values[:]

# modify predictor KMA with circle storms data
print('\nAdding Historical TCs to SLP_PREDICTOR KMA_RG bmus...')
pred.Mod_KMA_AddStorms(storm_dates, storm_categs)

