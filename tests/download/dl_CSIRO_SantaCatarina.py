#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import numpy as np
import xarray as xr

# tk libs
from lib.objs.tkpaths import PathControl
from lib.data_fetcher.CSIRO import Download_Gridded_Area, Download_Spec_Area

# --------------------------------------
# data storage and path control
pc = PathControl()
p_site = op.join(pc.DB.dwl.CSIRO, 'Brasil')



# --------------------------------------
# AREA 1 
area_name = 'SantaCatarina'
lonq = [304, 317]
latq = [-37, -24]
gridq = 'glob_24m'

p_area = op.join(p_site, area_name)
p_nc_grid = op.join(p_area, 'grid.nc')

# SPEC
#Download_Spec_Area(p_nc_spec, lonq, latq)

# GRIDDED
Download_Gridded_Area(p_nc_grid, lonq, latq, gridq)

