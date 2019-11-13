#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op

# pip
import numpy as np
import xarray as xr
from datetime import datetime
import pickle

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../../')

# teslakit
from teslakit.project_site import PathControl
from teslakit.io.matlab import ReadGowMat, ReadMatfile
from teslakit.intradaily import Calculate_Hydrographs
from teslakit.util.time_operations import datevec2datetime

# TODO: add hydrograph plots teslakit/plotting/intradaily

# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'Hydrographs')

# input files
p_KMA = op.join(p_test, 'KMA_daily_42_all.mat')
p_GOW = op.join(p_test, 'Waves_partitions', 'point1.mat')

# output files (hydrographs in dictionary or xarray.Datasets)
p_dbins = op.join(p_test, 'dbins_hydrographs.pk')
p_hydros = op.join(p_test, 'xds_hydrographs')


# --------------------------------------
# Load data

# load variables from matlab to xarray.Dataset
xds_WAVES = ReadGowMat(p_GOW)

d_KMA = ReadMatfile(p_KMA)
xds_BMUS = xr.Dataset(
    {
        'bmus': (('time',), d_KMA['KMA']['CorrectedBmus'])
    },
    coords = {
        'time': datevec2datetime(d_KMA['KMA']['Dates']),
    }
)


# --------------------------------------
# Calculate hydrographs 
dict_bins, list_xds_hydros = Calculate_Hydrographs(xds_BMUS, xds_WAVES)


# store hydrographs dictionary
pickle.dump(dict_bins, open(p_dbins,'wb'))
#pickle.load(open(p_dbins,'rb'))  # for loading

# store each hydrograph in one .nc file
if not op.isdir(p_hydros): os.makedirs(p_hydros)

for x in list_xds_hydros:
    fn = 'MUTAU_WT{0:02}.nc'.format(x.WT)
    x.to_netcdf(op.join(p_hydros, fn), 'w')

