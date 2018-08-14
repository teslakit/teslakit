#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import numpy as np
import xarray as xr

# tk libs
from lib.data_fetcher.CSIRO import  Download_Spec_Point

# data storage
p_data = op.join(op.dirname(__file__), '..', 'data')
p_test = op.join(p_data, 'tests', 'test_CSIRO')

p_site = op.join(p_test, 'AnaAgosto')
p_nc_grid = op.join(p_site, 'gridded.nc')


# --------------------------------------
# SPEC: Point list
lonp = [
    166.6,166.6,167.13,167.13,167.67,167.67,167.73,
    170.87,170.87,171.4,171.4,171.93,171.93,171.39
       ]
latp = [
    8.93,9.47,8.93,9.47,8.93,9.47,8.73,
    6.8,7.33,6.8,7.33,6.8,7.33,7.09
    ]

p_nc_spec = op.join(p_site, 'spec.nc')

# download point spec data
for lo,la in zip(lonp, latp):
    p_nc_spec = op.join(p_site, 'spec_lon_{0:5.2f}_lat_{1:4.2f}.nc'.format(lo,la))
    Download_Spec_Point(p_nc_spec, lo, la)


