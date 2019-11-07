#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op

# python libs
import numpy as np
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0, op.join(op.dirname(__file__), '..', '..', '..'))

# teslakit
from teslakit.climate_emulator import Climate_Emulator


# --------------------------------------
# Test data storage

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
p_test = op.join(p_data, 'tests', 'ClimateEmulator', 'CE_FitExtremes')

# input
p_KMA = op.join(p_test, 'xds_KMA.nc')
p_WVS_fam = op.join(p_test, 'xds_WVS_fam.nc')
p_WVS_pts = op.join(p_test, 'xds_WVS_pts.nc')

# output
p_ce = op.join(p_test, 'ce')  # climate emulator (fit)


# --------------------------------------
# Load data 

xds_KMA = xr.open_dataset(p_KMA)
xds_WVS_fam = xr.open_dataset(p_WVS_fam)
xds_WVS_pts = xr.open_dataset(p_WVS_pts)

print(xds_KMA)
print(xds_WVS_fam)
print(xds_WVS_pts)


# --------------------------------------
# Climate Emulator object 
CE = Climate_Emulator(p_ce)


# Fit climate emulator and save 
config = {
    'name_fams': ['sea', 'swell_1', 'swell_2'],
    'force_empirical': ['sea_Tp'],
}
CE.FitExtremes(xds_KMA, xds_WVS_pts, xds_WVS_fam, config)

print('test done.')

