#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import numpy as np
import xarray as xr

# tk libs
from lib.objs.tkpaths import PathControl
from lib.objs.predictor import Predictor
from lib.tcyclone import Extract_Circle


# --------------------------------------
# data storage and path control
pc = PathControl()
pc.SetSite('test_site')


# --------------------------------------
# load TCs and select inside circle
xds_wmo_fix = xr.open_dataset(pc.DB.tcs.noaa_fix)

p_lon = 178
p_lat = -17.5
r = 4

_, xds_in = Extract_Circle(
    xds_wmo_fix, p_lon, p_lat, r)

storm_dates = xds_in.dmin_date.values[:]
storm_categs = xds_in.category.values[:]


# --------------------------------------
# Load tesla-kit predictor
pred = Predictor(pc.site.est.pred_slp)
pred.Load()

# modify predictor KMA with circle storms data
pred.Mod_KMA_AddStorms(storm_dates, storm_categs)

print pred.KMA.sorted_bmus_storms

