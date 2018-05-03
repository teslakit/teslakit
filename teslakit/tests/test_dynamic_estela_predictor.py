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
from lib.io.matlab import ReadMatfile
from lib.predictor import dynamic_estela_predictor

# data storage
p_data = op.join(op.dirname(__file__),'..','data')

p_test = op.join(p_data, 'test_dynamic_estela_predictor')
p_xds_slp = op.join(p_test, 'xds_SLP_day.nc')
p_xds_estela = op.join(p_test, 'xds_estela.nc')
p_xds_estela_pred = op.join(p_test, 'xds_estela_pred.nc')


# --------------------------------------
# load test data
xds_SLP_day = xr.open_dataset(p_xds_slp)
xds_est_site = xr.open_dataset(p_xds_estela)

# Generate estela predictor
xds_estela_SLP = dynamic_estela_predictor(
    xds_SLP_day, 'SLP', xds_est_site.D_y1993to2012.values)
xds_estela_SLP.to_netcdf(p_xds_estela_pred)


# plot data
xds_estela_SLP.isel(time=0).SLP_comp.plot()
plt.show()

