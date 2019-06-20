#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..','..'))

# pip
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# tk libs
from teslakit.project_site import PathControl
from teslakit.io.matlab import ReadMatfile
from teslakit.estela import dynamic_estela_predictor


# TODO: REVISAR TEST

# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ESTELA', 'test_dynamic_estela_predictor')

p_xds_slpday = op.join(p_test, 'xds_SLP_day.nc')  # extracted SLP
p_xds_estela = op.join(p_test, 'xds_estela_pred.nc')


# --------------------------------------
# load test data
xds_SLP_day = xr.open_dataset(p_xds_slpday)
xds_est_site = xr.open_dataset(p_xds_estela)
print(xds_est_site)

# Generate estela predictor
xds_estela_SLP = dynamic_estela_predictor(
    xds_SLP_day, 'SLP', xds_est_site.D_y1993to2012.values)
xds_estela_SLP.to_netcdf(p_xds_estela_pred)
print(xds_estela_SLP)

