#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op

# pip 
import numpy as np
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0, op.join(op.dirname(__file__), '..', '..', '..'))

# teslakit
from teslakit.database import Database
from teslakit.waves import GetDistribution


# --------------------------------------
# Test data storage

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('ROI')


# waves partitions data
WVS_pts = db.Load_WAVES_partitions()
print(WVS_pts)

# wave families sectors
fams_sectors = [(210, 22.5), (22.5, 135)]


# calculate waves families
WVS_fams = GetDistribution(WVS_pts, fams_sectors)
print(WVS_fams)

