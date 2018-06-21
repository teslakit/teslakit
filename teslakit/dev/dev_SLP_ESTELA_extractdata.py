#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# tk libs
from lib.io.cfs import ReadSLP

# data storage
p_data = op.join(op.dirname(__file__), '..', 'data')
p_test = op.join(p_data, 'tests', 'tests_estela', 'Roi_Kwajalein')

# SLP CFS database
p_pred_SLP = op.join(p_data, 'CFS', 'prmsl')


# teslakit files
p_SLP_data = op.join(p_test, 'SLP.nc')  # to store extracted SLP

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

