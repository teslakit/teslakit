#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import sys

from datetime import datetime, timedelta

# pip
import numpy as np
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0, op.join(op.dirname(__file__), '..', '..', '..'))

# teslakit
from teslakit.database import Database
from teslakit.alr import ALR_WRP


# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')

# alr test
#p_alrw = db.paths.site.SST.alrw
p_alrw = db.paths.site.MJO.alrw
#p_alrw = db.paths.site.ESTELA.alrw

# ALR wrap
ALRW = ALR_WRP(p_alrw)


# load model 
ALRW.LoadModel()
ALRW.LoadBmus_Fit()

# simulation 
num_sims = 1
d1_sim = np.datetime64('2260-01-01').astype(datetime)
d2_sim = np.datetime64('2265-01-01').astype(datetime)
dates_sim = [d1_sim + timedelta(days=i) for i in range((d2_sim-d1_sim).days+1)]

# launch simulation and check progress_bar behaviour
xds_alr = ALRW.Simulate(num_sims, dates_sim)
print(xds_alr)
print()
xds2 = xr.open_dataset(op.join(db.paths.site.MJO.alrw, 'xds_output.nc'))
print(xds2)
