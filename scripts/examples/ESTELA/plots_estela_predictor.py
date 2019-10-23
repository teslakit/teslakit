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
from teslakit.estela import Predictor
from teslakit.plotting.estela import Plot_ESTELA


# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')

# estela predictor
xds_est = db.Load_ESTELA_data()

# plot ESTELA (basemap)
Plot_ESTELA(
    xds_est.pnt_longitude, xds_est.pnt_latitude,
    xds_est.F_y1993to2012 * xds_est.mask_e95,
    xds_est.D_y1993to2012 * xds_est.mask_e95,
    lon1=110, lon2=290, lat1=-50, lat2=70,
)


# load predictor
pred = Predictor(db.paths.site.ESTELA.pred_slp)
pred.Load()

# test ESTELA PCA EOFs plot
pred.Plot_EOFs_EstelaPred()

# Plot PCs 2D with DWTs centroids
pred.Plot_DWT_PCs(n=6)

# test DWTs mean plot
pred.Plot_DWTs('SLP', kind='mean', show=True)

# test DWTs anomally plot
pred.Plot_DWTs('SLP', kind='anom', show=True)

# test DWTs probs plot
pred.Plot_DWTs_Probs()

# test DWTs PCs 3D plot
pred.Plot_PCs_3D()



