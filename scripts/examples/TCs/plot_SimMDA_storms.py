#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import sys

# pip
import numpy as np
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0, op.join(op.dirname(__file__), '..', '..', '..'))

# teslakit
from teslakit.database import Database
from teslakit.plotting.storms import Plot_Params_Hist_vs_Sim_scatter, \
Plot_Params_Hist_vs_Sim_histogram


# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')


# Load storms parameters real and syntethic 
_,xds_TCs_r2_params = db.Load_TCs_r2()
xds_TCs_r2_sim_params = db.Load_TCs_r2_sim_params()


# Plot storms tracks and storm parametrized inside radius 
Plot_Params_Hist_vs_Sim_scatter(xds_TCs_r2_params, xds_TCs_r2_sim_params)

# Plot storms tracks and storm parametrized inside radius 
Plot_Params_Hist_vs_Sim_histogram(xds_TCs_r2_params, xds_TCs_r2_sim_params)
