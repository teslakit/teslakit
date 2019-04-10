#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# pip
import numpy as np
import xarray as xr

# tk dl 
from teslakit.project_site import PathControl
from teslakit_downloader.CSIRO import Download_Gridded_Area, Download_Spec_Point

# --------------------------------------
# data storage and path control
pc = PathControl()
p_tests = pc.p_test_data
p_site = op.join(p_tests, 'CSIRO', 'MAJURO')


# --------------------------------------
# AREA 1 
area_name = 'A1'
lonq = [166.5, 168]
latq = [8.5, 9.8]

p_area = op.join(p_site, area_name)
p_nc_spec = op.join(p_area, 'spec.nc')
p_nc_grid = op.join(p_area, 'grid.nc')

# SPEC
Download_Spec_Area(p_nc_spec, lonq, latq)

# GRIDDED
Download_Gridded_Area(p_nc_grid, lonq, latq, 'pac_4m')


# --------------------------------------
# AREA 2 
area_name = 'A2'
lonq = [170.5, 172]
latq = [6.5, 7.5]

p_area = op.join(p_site, area_name)
p_nc_spec = op.join(p_area, 'spec.nc')
p_nc_grid = op.join(p_area, 'grid.nc')

# SPEC
Download_Spec_Area(p_nc_spec, lonq, latq)

# GRIDDED
Download_Gridded_Area(p_nc_grid, lonq, latq, 'pac_4m')

