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
sys.path.insert(0,'../../../')

# teslakit
from teslakit.project_site import PathControl
from teslakit.waves import TWL_WavesFamilies, TWL_AnnualMaxima
from teslakit.io.matlab import ReadMatfile
from teslakit.util.time_operations import DateConverter_Mat2Py as dmp


# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ClimateEmulator', 'ml_annualmaxima')


# --------------------------------------
# load test data (matlab ) 

p_sim = op.join(p_test, 'test_sim.mat')
dm_sim = ReadMatfile(p_sim)['dmat']
sim_data = dm_sim['Sim']
time_dnum = dm_sim['datenum']

# mount compatible xarray.Dataset
xds_wvs_sim = xr.Dataset(
    {
        'sea_Hs'      : (('time',), sim_data[:,0]),
        'sea_Tp'      : (('time',), sim_data[:,1]),
        'sea_Dir'     : (('time',), sim_data[:,2]),
        'swell_1_Hs'  : (('time',), sim_data[:,3]),
        'swell_1_Tp'  : (('time',), sim_data[:,4]),
        'swell_1_Dir' : (('time',), sim_data[:,5]),
        'swell_2_Hs'  : (('time',), sim_data[:,6]),
        'swell_2_Tp'  : (('time',), sim_data[:,7]),
        'swell_2_Dir' : (('time',), sim_data[:,8]),
    },
    coords = {'time' : dmp(time_dnum)}
)



# --------------------------------------
# Calculate TWL for waves families data 

xds_TWL = TWL_WavesFamilies(xds_wvs_sim)
print(xds_TWL)
print()


# Calculate annual maxima (manually: time index not monotonic)
xds_TWL_AnMax = TWL_AnnualMaxima(xds_TWL)
print(xds_TWL_AnMax)

