#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..','..'))

# pip
import numpy as np

# tk dl
from teslakit.project_site import PathControl
from teslakit_downloader.CSIRO import Download_Gridded_Area, Download_Spec_Point


# --------------------------------------
# test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'CSIRO', 'SantaCatarina')

# downloaded files 
p_nc_grid = op.join(p_test, 'gridded.nc')
p_nc_spec = op.join(p_test, 'spec.nc')


# --------------------------------------
# Point
lonq = [304, 317]
latq = [-37, -24]

# download point gridded data
xds_p_grid = Download_Gridded_Area(p_nc_grid, lonq, latq)
print(xds_p_grid)

# download point spec data
xds_p_spec = Download_Spec_Area(p_nc_spec, lonq, latq)
print(xds_p_spec)

