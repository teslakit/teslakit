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
from teslakit.plotting.awt import Plot_AWT_Validation, Plot_AWTs, \
Plot_AWTs_Dates, Plot_AWT_PCs_3D



# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')


# Load SST AWT KMA and PCA 
xds_PCA = db.Load_SST_PCA()
xds_KMA = db.Load_SST_KMA()

# load PCs simulated with copula
d_PCs_fit, d_PCs_rnd = db.Load_SST_PCs_fit_rnd()

# Plot PCs 3D
Plot_AWT_PCs_3D(d_PCs_fit, d_PCs_rnd)

# Plot AWTs
Plot_AWTs(xds_KMA, xds_PCA.pred_lon)

# Plot AWTs dates
Plot_AWTs_Dates(xds_KMA)

# Plot AWT Validation report
Plot_AWT_Validation(
    xds_KMA, xds_PCA.pred_lon,
    d_PCs_fit, d_PCs_rnd
)

