#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys
import time
sys.path.insert(0, op.join(op.dirname(__file__),'..','..'))

# python libs
import numpy as np
import xarray as xr

# custom libs
from teslakit.project_site import PathControl
from teslakit.extremes import FitGEV_KMA_Frechet

# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ClimateEmulator', 'gev_fit_kma_fretchet')

# input
p_npz = op.join(p_test, 'swell_1_Hs.npz')

# --------------------------------------
# Load data 
npzf = np.load(p_npz)
bmus = npzf['arr_0']
n_clusters = npzf['arr_1']
var_wvs = npzf['arr_2']


print(bmus)
print(n_clusters)
print(var_wvs)
print()

# TODO: small differences with ML at nlogl_1-nlogl_2 = 1.92
gp_pars = FitGEV_KMA_Frechet(
    bmus, n_clusters, var_wvs)

print(gp_pars)
