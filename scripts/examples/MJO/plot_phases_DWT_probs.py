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
from teslakit.plotting.wts import Plot_Probs_WT_WT


# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')

MJO_ncs = 8
DWT_ncs = 36

# MJO, DWTs historical data
xds_MJO_hist, xds_DWT_hist = db.Load_MJO_DWTs_Plots_hist()
MJO_phase = xds_MJO_hist.phase.values[:]
DWT_bmus = xds_DWT_hist.bmus.values[:]

Plot_Probs_WT_WT(
    MJO_phase, DWT_bmus, MJO_ncs, DWT_ncs,
    wt_colors=False, ttl='MJO Phases / DWT bmus (Historical)')


# MJO, DWTs simulated data
#xds_MJO_sim, xds_DWT_sim = db.Load_MJO_DWTs_Plots_sim()
#MJO_phase = xds_MJO_sim.phase.values[:] - 1 # start at 0
#DWT_bmus = xds_DWT_sim.bmus.values[:]

#print(np.unique(MJO_phase))
#print(np.unique(DWT_bmus))

# TODO simulated phase not classified

#Plot_Probs_WT_WT(
#    MJO_phase, DWT_bmus, MJO_ncs, DWT_ncs,
#    wt_colors=False, ttl='MJO Phases / DWT bmus (Historical)')

