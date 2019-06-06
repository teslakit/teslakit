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
from teslakit.extremes import Smooth_GEV_Shape

# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ClimateEmulator', 'opt_smooth_shape')

# input
p_npz = op.join(p_test, 'swell_1_Hs.npz')

# --------------------------------------
# Load data 
npzf = np.load(p_npz)
cenEOFs = npzf['arr_0']
sh_GEV = npzf['arr_1']
print(sh_GEV)


# test original function
t0 = time.time()
shape_smooth1 = Smooth_GEV_Shape(cenEOFs, sh_GEV)
dt_test1 = time.time()-t0

print('test 1')
print(dt_test1)
print(shape_smooth1)
print()

