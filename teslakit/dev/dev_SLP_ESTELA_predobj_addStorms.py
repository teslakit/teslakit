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
from lib.objs.predictor import Predictor
from lib.hurricanes import Extract_Circle

# data storage
p_data = op.join(op.dirname(__file__), '..', 'data')
p_data_hurr = op.join(p_data, 'HURR')

p_hurr_noaa_fix = op.join(p_data_hurr, 'Allstorms.ibtracs_wmo.v03r10_fix.nc')
p_test = op.join(p_data, 'tests', 'tests_estela', 'Roi_Kwajalein')


# --------------------------------------
# load storms and select inside circle
xds_wmo_fix = xr.open_dataset(p_hurr_noaa_fix)

p_lon = 178
p_lat = -17.5
r = 4

xds_storms_r, xds_inside = Extract_Circle(
    xds_wmo_fix, p_lon, p_lat, r)

storm_dates = xds_inside.inside_date.values[:]
storm_categs = xds_inside.inside_category.values[:]


# --------------------------------------
# Load tesla-kit predictor
p_SLP_pred = op.join(p_test, 'pred_SLP')
pred = Predictor(p_SLP_pred)
pred.Load()

# modify predictor KMA with circle storms data
pred.Mod_KMA_AddStorms(storm_dates, storm_categs)
pred.Save()

print pred.KMA.sorted_bmus_storms

