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
from lib.io.matlab import ReadMatfile
from lib.predictor import spatial_gradient

# data storage
p_data = op.join(op.dirname(__file__),'..','data')

p_mat_mg = op.join(p_data, 'tests_spatial_gradient', 'meshgrid.mat')
p_mat_slp = op.join(p_data, 'tests_spatial_gradient', 'slp.mat')


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

print xds_SLP

