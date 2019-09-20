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


# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')

# estela predictor
pred = Predictor(db.paths.site.ESTELA.pred_slp)
pred.Load()

# test ESTELA PCA EOFs plot
#pred.Plot_EOFs_EstelaPred()

# test DWTs mean plot
pred.Plot_DWTs('SLP', mask='mask_estela')

