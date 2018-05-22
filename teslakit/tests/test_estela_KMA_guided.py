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
from lib.io.matlab import ReadMatfile, ReadGowMat
from lib.KMA import KMA_regression_guided as KMA_RG
from lib.KMA import SimpleMultivariateRegressionModel as SMRM

# data storage
p_data = op.join(op.dirname(__file__),'..','data')
p_test = op.join(p_data, 'tests_estela_PCA')

p_PCA = op.join(p_test, 'xds_SLP_PCA.nc')
p_GOW = op.join(p_test, 'gow2_062_ 9.50_167.25.mat')

# TODO: CAMBIAR GOW A WAVES POR GENERALIZAR

# load PCA and GOW WAVES data
xds_PCA = xr.open_dataset(p_PCA)
xds_GOW = ReadGowMat(p_GOW)

# calculate Fe
hs = xds_GOW.hs
tm = xds_GOW.t02
Fe = np.multiply(hs**2,tm)**(1.0/3)
xds_GOW.update({
    'Fe':(('time',), Fe)
})

# select time window and do data daily mean
xds_GOW = xds_GOW.sel(
    time=slice('1979-01-22','1980-12-31')
).resample(time='1D').mean()


# calculate regresion model between predictand and predictor
name_vars = ['hs', 't02', 'Fe']
xds_Yregres = SMRM(xds_PCA, xds_GOW, name_vars)


# classification: KMA regresion guided
num_clusters = 36
xds_KMA = KMA_RG(xds_PCA, xds_Yregres, num_clusters)


# todo: PLOTEAR CLASIFICACION Y RESUÑTADPS KMA
