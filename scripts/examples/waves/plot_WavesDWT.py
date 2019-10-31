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
from teslakit.custom_dateutils import xds_common_dates_daily as xcd_daily
from teslakit.plotting.waves import Plot_Waves_DWTs


# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')


# load waves families (full data)
xds_wvs_fams = db.Load_WAVES_fams()

# load predictor
pred = Predictor(db.paths.site.ESTELA.pred_slp)
pred.Load()

# DWTs bmus
xds_DWTs = pred.KMA
xds_BMUS = xr.Dataset(
    {
        'bmus':(('time',), xds_DWTs['sorted_bmus_storms'].values[:])
    },
    coords = {'time': xds_DWTs.time.values[:]}
)
n_clusters = 42


# common dates
dates_common= xcd_daily([xds_wvs_fams, xds_DWTs])

# waves at common dates
xds_wvs_fams_sel = xds_wvs_fams.sel(time=dates_common)

# bmus at common dates
bmus = xds_BMUS.sel(time=dates_common).bmus.values[:]


# Plot Waves Families by DWTs
Plot_Waves_DWTs(xds_wvs_fams_sel, bmus, n_clusters)

