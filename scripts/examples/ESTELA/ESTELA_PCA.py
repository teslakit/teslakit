#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op

# pip
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../../')

# teslakit
from teslakit.project_site import PathControl
from teslakit.io.matlab import ReadMatfile
from teslakit.PCA import CalcPCA_EstelaPred
from teslakit.plotting.EOFs import Plot_EOFs_EstelaPred


# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ESTELA', 'test_estela_PCA')


# --------------------------------------
# use teslakit test data 
p_estela_pred = op.join(p_test, 'xds_SLP_estela_pred.nc')
xds_SLP_estela_pred = xr.open_dataset(p_estela_pred)

# Calculate PCA
xds_PCA = CalcPCA_EstelaPred(xds_SLP_estela_pred, 'SLP')
xds_PCA.to_netcdf(op.join(p_test, 'xds_SLP_PCA.nc'))
print(xds_PCA)

# Plot EOFs
n_plot = 3
p_save = op.join(p_test, 'Plot_EOFs_EstelaPred')
Plot_EOFs_EstelaPred(xds_PCA, n_plot, p_save)


