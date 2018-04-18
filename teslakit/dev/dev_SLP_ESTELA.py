#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import numpy as np
import matplotlib.pyplot as plt

# tk libs
from lib.io.matlab import ReadGowMat
from lib.io.cfs import ReadSLP
from lib.predictor import spatial_gradient

# data storage
p_data = op.join(op.dirname(__file__),'..','data')
p_test = op.join(p_data, 'tests_estela', 'Roi_Kwajalein')

p_estela = op.join(p_test, 'kwajalein_roi_obj.mat')         # mask with estela
p_pred_SLP = op.join(p_data,'tests_estela','CFS','prmsl')   # SLP predictor
p_gowpoint = op.join(p_test, 'gow2_062_ 9.50_167.25.mat')   # gow point data
p_coast = op.join(p_test, 'Costa.mat')                      # coast 

p_results = op.join(p_test, 'out_KWAJALEIN')


# --------------------------------------
# load waves data

# load from .mat gow file
#xds_gow = ReadGowMat(p_gowpoint)


# --------------------------------------
# load predictor data (we use SLP and SLP gradients)
xds_SLP = ReadSLP(p_pred_SLP)
xds_SLP.rename({'PRMSL_L101':'SLP'}, inplace=True)


# site coordinates 
lat1 = 60.5
lat2 = 0
lon1 = 115
lon2 = 280

# cut data and resample to 2º lon,lat
xds_SLP_site = xds_SLP.sel(
    latitude = slice(lat1, lat2, 4),
    longitude = slice(lon1, lon2, 4),
)

print xds_SLP_site


# parse data to daily average 
xds_SLP_day = xds_SLP_site.resample(time='1D').mean()

# calculate daily gradients
xds_SLP_day = spatial_gradient(xds_SLP_day, 'SLP')
print xds_SLP_day




# TODO: DESPUES GENERAR Y APLICAR MASCARA COSTA Y ESTELA



