#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..','..'))

# pip
import numpy as np
import xarray as xr

# tk
from teslakit.project_site import PathControl
from teslakit.io.matlab import ReadMatfile, ReadGowMat
from teslakit.KMA import KMA_regression_guided
from teslakit.KMA import SimpleMultivariateRegressionModel as SMRM


# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ESTELA', 'test_estela_PCA')

# input
p_PCA = op.join(p_test, 'xds_SLP_PCA.nc')
p_WAVES = op.join(p_test, 'gow2_062_ 9.50_167.25.mat')

# output
p_KMA_save = op.join(p_test, 'xds_test_KMArg.nc')  # to save test results


# --------------------------------------
# load PCA and GOW WAVES data
xds_PCA = xr.open_dataset(p_PCA)
xds_WAVES = ReadGowMat(p_WAVES)

# calculate Fe
hs = xds_WAVES.hs
tm = xds_WAVES.t02
Fe = np.multiply(hs**2,tm)**(1.0/3)
xds_WAVES.update({
    'Fe':(('time',), Fe)
})

# select time window and do data daily mean
xds_WAVES = xds_WAVES.sel(
    time=slice('1979-01-22','1980-12-31')
).resample(time='1D').mean()


# calculate regresion model between predictand and predictor
name_vars = ['hs', 't02', 'Fe']
xds_Yregres = SMRM(xds_PCA, xds_WAVES, name_vars)


# classification: KMA regresion guided
num_clusters = 36
repres = 0.95
alpha = 0.3
xds_KMA = KMA_regression_guided(
    xds_PCA, xds_Yregres, num_clusters, repres, alpha)
xds_KMA.to_netcdf(p_KMA_save)

print(xds_KMA)

