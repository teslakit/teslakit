#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op

# pip
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../../')

# teslakit
from teslakit.project_site import PathControl
from teslakit.io.matlab import ReadMatfile
from teslakit.estela import spatial_gradient

# TODO: RESIVAR DATOS TEST

# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ESTELA', 'test_spatial_gradient')

# input
p_mat_mg = op.join(p_test, 'meshgrid.mat')
p_mat_slp = op.join(p_test, 'slp.mat')


# --------------------------------------
# load test data
dmat_mg = ReadMatfile(p_mat_mg)
dmat_slp = ReadMatfile(p_mat_slp)

XR = dmat_mg['XR1']
YR = dmat_mg['YR1']
SLP = dmat_slp['ppp']

longitude = XR[1,:]
latitude = YR[:,1]

# set xr.Dataset
xds_SLP = xr.Dataset(
    {
        'SLP': (('time','latitude','longitude'), SLP),
    },
    coords = {
        'time': range(SLP.shape[0]),
        'latitude': latitude,
        'longitude': longitude,
    }
)


# calculate daily gradients
xds_SLP = spatial_gradient(xds_SLP, 'SLP')

print(xds_SLP)

