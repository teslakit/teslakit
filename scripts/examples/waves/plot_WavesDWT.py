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
from teslakit.plotting.waves import Plot_Waves_DWTs


# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')


# load waves
xds_wvs_fams = db.Load_WAVES_fams_noTCs()

# load predictor
pred = Predictor(db.paths.site.ESTELA.pred_slp)
pred.Load()

# DWTs
xds_DWTs = pred.KMA


# Plot Waves Families by DWTs
Plot_Waves_DWTs(xds_wvs_fams, xds_DWTs)

