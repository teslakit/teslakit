#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# tk libs
from lib.io.matlab import ReadGowMat, ReadCoastMat, ReadEstelaMat
from lib.io.cfs import ReadSLP
from lib.estela import spatial_gradient, mask_from_poly, \
dynamic_estela_predictor
from lib.PCA import CalcPCA_EstelaPred as CalcPCA

# data storage
p_data = op.join(op.dirname(__file__),'..','data')
p_test = op.join(p_data, 'tests_estela', 'Roi_Kwajalein')

p_estela_mat = op.join(p_test, 'kwajalein_roi_obj.mat')         # mask with estela
p_pred_SLP = op.join(p_data,'tests_estela','CFS','prmsl')   # SLP predictor
p_gowpoint = op.join(p_test, 'gow2_062_ 9.50_167.25.mat')   # gow point data
p_coast_mat = op.join(p_test, 'Costa.mat')                      # coast 

p_results = op.join(p_test, 'out_KWAJALEIN')

p_SLP_save = op.join(p_test, 'SLP.nc')


# --------------------------------------
# load sea polygons for mask generation
ls_sea_poly = ReadCoastMat(p_coast_mat)


# --------------------------------------
# load estela data 
xds_est = ReadEstelaMat(p_estela_mat)


# --------------------------------------
# load waves data from .mat gow file
xds_GOW = ReadGowMat(p_gowpoint)


# --------------------------------------
# site coordinates 
lat1 = 60.5
lat2 = 0.5
lon1 = 115
lon2 = 279
resample = 4  #2º

# load predictor data (SLP) from CFSR and save to .nc 
xds_SLP_site = ReadSLP(
    p_pred_SLP,
    lat1, lat2, lon1, lon2, resample,
    p_save=p_SLP_save)


# --------------------------------------
# load and use xarray saved predictor data (faster)
xds_SLP_site = xr.open_dataset(p_SLP_save)

# parse data to daily average 
xds_SLP_day = xds_SLP_site.resample(time='1D').mean()

# calculate daily gradients
# TODO: ADD ONE ROW/COL EACH SIDE
xds_SLP_day = spatial_gradient(xds_SLP_day, 'SLP')

# generate land mask with land polygons 
xds_SLP_day = mask_from_poly(xds_SLP_day, ls_sea_poly, 'mask_land')


# resample estela to site mesh
xds_est_site = xds_est.sel(
    longitude = xds_SLP_day.longitude,
    latitude = xds_SLP_day.latitude,
)

# generate mask using estela
mask_est = np.zeros(xds_est_site.D_y1993to2012.shape)
mask_est[np.where(xds_est_site.D_y1993to2012<1000000000)]=1

xds_SLP_day.update({
    'mask_estela':(('latitude','longitude'), mask_est)
})


# Generate estela predictor
xds_SLP_estela_pred = dynamic_estela_predictor(
    xds_SLP_day, 'SLP', xds_est_site.D_y1993to2012)

# TODO: EXISTEN DIFERENCIAS ENTRE PREDICTOR MATLAB Y ESTE. 
# REPASAR MONTAJE PREDICTOR


# Calculate PCA
xds_PCA = CalcPCA(xds_SLP_estela_pred, 'SLP')


# calculate Fe (from GOW waves data)
hs = xds_GOW.hs
tm = xds_GOW.t02
Fe = np.multiply(hs**2,tm)**(1.0/3)
xds_GOW.update({
    'Fe':(('time',), Fe)
})

# select time window and do data daily mean
xds_GOW = xds_GOW.sel(
    time=slice('1979-01-22','1980-12-31')
).resample(time='1D').mean()


# calculate regresion model between predictand and predictor
name_vars = ['hs', 't02', 'Fe']
xds_Yregres = SMRM(xds_PCA, xds_GOW, name_vars)

