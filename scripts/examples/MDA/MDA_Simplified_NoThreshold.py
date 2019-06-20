#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op

# pip
import xarray as xr
import numpy as np

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../../')

# teslakit
from teslakit.project_site import PathControl
from teslakit.mda import MaxDiss_Simplified_NoThreshold


# --------------------------------------
# test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'MDA')

# input
p_input = op.join(p_test, 'TCs_sim_r2_params.nc')

# output
p_output = op.join(p_test, 'TCs_MDA_params.nc')


# --------------------------------------
# load input data
xds_input = xr.open_dataset(p_input)
print(xds_input)

# input: 100000 storm parameters
pmean = xds_input.pressure_mean.values[:]
pmin = xds_input.pressure_min.values[:]
gamma = xds_input.gamma.values[:]
delta = xds_input.delta.values[:]
vmean = xds_input.velocity_mean.values[:]


# use MDA to get a small subset of cases (1000)
data_mda = np.column_stack((pmean, pmin, vmean, delta, gamma))
ix_scalar = [0,1,2]  # scalar variables indexes (pmean, pmin, vmean)
ix_directional = [3,4]  # directional variables indexes (delta, gamma)
num_sel_mda = 1000  # num of cases to select using MDA

centroids = MaxDiss_Simplified_NoThreshold(
    data_mda, num_sel_mda, ix_scalar, ix_directional
)


# store MDA output
xds_output = xr.Dataset(
    {
        'pressure_mean':(('storm'), centroids[:,0]),
        'pressure_min':(('storm'), centroids[:,1]),
        'gamma':(('storm'), centroids[:,2]),
        'delta':(('storm'), centroids[:,3]),
        'velocity_mean':(('storm'), centroids[:,4]),
    },
    coords = {
        'storm':(('storm'), np.arange(num_sel_mda))
    },
)
print(xds_output)
xds_output.to_netcdf(p_output)

