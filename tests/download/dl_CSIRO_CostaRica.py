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
p_site = op.join(pc.DB.dwl.CSIRO, 'CostaRica')



# --------------------------------------
# Point 
#8º 44' 48.8'' N, 85º 17' 7.5'' W
p_name = 'p1'
latq = [8.746889]
lonq = [275.7312]
gridq = 'glob_24m'

p_nc_grid = op.join(p_site, 'grid_{0}.nc'.format(p_name))

# SPEC
#Download_Spec_Area(p_nc_spec, lonq, latq)

# GRIDDED
Download_Gridded_Area(p_nc_grid, lonq, latq, gridq)

