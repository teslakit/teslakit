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
from teslakit.util.operations import GetRepeatedValues



# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')


# --------------------------------------
# load data and set parameters
p_alr_output = op.join(db.paths.site.MJO.alrw, 'xds_output.nc')

# get bmus series
xds_out = xr.open_dataset(p_alr_output)
evbmus_sim = xds_out.isel(n_sim=0).evbmus_sims.values[:]

# generate random array to order
vals = np.random.randint(100, size=evbmus_sim.shape)


# now find subsequences of repeated adyacent bmus
l_ad = GetRepeatedValues(evbmus_sim)
print(len(evbmus_sim))
print(len(l_ad))
print()

for s,e in l_ad:
    print(s, e)

    # get sort index 
    ixs = np.argsort(vals[s:e])

    print(vals[s:e])
    vals[s:e] = vals[s:e][ixs]
    print(vals[s:e])
    print()

    sys.exit()
