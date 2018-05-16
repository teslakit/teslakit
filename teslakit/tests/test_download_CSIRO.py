#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import numpy as np
import xarray as xr

# tk libs
from lib.data_fetcher import Download_CSIRO

# data storage
p_data = op.join(op.dirname(__file__),'..','data')

p_test = op.join(p_data, 'test_CSIRO')
p_nc_point = op.join(p_test, 'csiro_down_point.nc')
p_nc_mesh = op.join(p_test, 'csiro_down_mesh.nc')


# --------------------------------------
# parameters 
var_names = ['hs2', 'tp0']


# download point data
lonp = [171]
latp = [7.5]
#xds_p = Download_CSIRO(p_nc_point, lonp, latp, var_names)
#print xds_p


# download mesh data 
lonm = [170, 173]
latm = [7, 8]
xds_m = Download_CSIRO(p_nc_mesh, lonm, latm, var_names)
print xds_m

