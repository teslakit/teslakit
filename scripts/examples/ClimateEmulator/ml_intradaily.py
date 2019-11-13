#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys

# python libs
import numpy as np
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../../')

# teslakit
from teslakit.project_site import PathControl
from teslakit.io.matlab import ReadMatfile
from teslakit.util.time_operations import DateConverter_Mat2Py as dmp
from teslakit.waves import Aggregate_WavesFamilies, Intradaily_Hydrograph


# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ClimateEmulator', 'ml_intradaily')


# --------------------------------------
# load test data (matlab) 

# load simulated waves fams and mu/tau (TCs)
p_sim_wvs = op.join(p_test, 'output_sim.mat')
dm_sim = ReadMatfile(p_sim_wvs)

sim_wvs_fams = dm_sim['Sim']      # waves 
sim_tcs_mu   = dm_sim['SIM_MU']   # tcs 
sim_tcs_tau  = dm_sim['SIM_TAU']  # tcs
sim_tcs_ss   = dm_sim['SIM_SS']   # tcs

# get time from aux file
p_aux = op.join(p_test, 'aux.mat')
dm_aux = ReadMatfile(p_aux)['dmat']
time_dnum = dmp(dm_aux['datenum'])

# xarray.Dataset for simulated waves
xds_wvs_sim = xr.Dataset(
    {
        'sea_Hs'      : (('time',), sim_wvs_fams[:,0]),
        'sea_Tp'      : (('time',), sim_wvs_fams[:,1]),
        'sea_Dir'     : (('time',), sim_wvs_fams[:,2]),
        'swell_1_Hs'  : (('time',), sim_wvs_fams[:,3]),
        'swell_1_Tp'  : (('time',), sim_wvs_fams[:,4]),
        'swell_1_Dir' : (('time',), sim_wvs_fams[:,5]),
        'swell_2_Hs'  : (('time',), sim_wvs_fams[:,6]),
        'swell_2_Tp'  : (('time',), sim_wvs_fams[:,7]),
        'swell_2_Dir' : (('time',), sim_wvs_fams[:,8]),
    },
    coords = {'time' : time_dnum}
)

# xarray.Dataset for simulated TCs
sim_tcs_mu [sim_tcs_mu < 0.5] = 0.5
xds_tcs_sim = xr.Dataset(
    {
        'mu'      : (('time',), sim_tcs_mu),
        'tau'      : (('time',), sim_tcs_tau),
        'ss'      : (('time',), sim_tcs_ss),
    },
    coords = {'time' : time_dnum}
)


print(xds_wvs_sim)
print()
print(xds_tcs_sim)
print()


# --------------------------------------

# Aggregate waves families data 
xds_wvs_agr = Aggregate_WavesFamilies(xds_wvs_sim)
print(xds_wvs_agr)
print()

# calculate intradaily hydrographs
xds_hg = Intradaily_Hydrograph(xds_wvs_agr, xds_tcs_sim)
print(xds_hg)
print()


