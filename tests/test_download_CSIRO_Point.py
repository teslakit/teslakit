#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import numpy as np
import xarray as xr

# tk libs
from lib.data_fetcher.CSIRO import Download_Gridded, Download_Spec

# data storage
p_data = op.join(op.dirname(__file__), '..', 'data')
p_test = op.join(p_data, 'tests', 'test_CSIRO')

p_site = op.join(p_test, 'Fortaleza')
p_nc_grid = op.join(p_site, 'gridded.nc')
p_nc_spec = op.join(p_site, 'spec.nc')


# --------------------------------------
# Point
lonp = [321.52]
latp = [-3.68]

# download point gridded data
xds_p_grid = Download_Gridded(p_nc_grid, lonp, latp)
print(xds_p_grid)

# download point spec data
xds_p_spec = Download_Spec(p_nc_spec, lonp, latp)
print(xds_p_spec)

