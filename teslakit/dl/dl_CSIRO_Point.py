#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
pf = op.join(op.dirname(__file__),'..')
sys.path.insert(0, pf)

import numpy as np

# custom libs
from lib.data_fetcher.CSIRO import Download_Gridded_Area, Download_Spec_Point

# --------------------------------------
# data storage
p_site = op.join(pf, 'data', 'CSIRO', 'PointTest')
p_nc_grid = op.join(p_site, 'gridded.nc')
p_nc_spec = op.join(p_site, 'spec.nc')


# --------------------------------------
# Point
lonp = [356.22]
latp = [43.89]

# download point gridded data
xds_p_grid = Download_Gridded_Area(p_nc_grid, lonp, latp)
print(xds_p_grid)

# download point spec data
#xds_p_spec = Download_Spec_Area(p_nc_spec, lonp, latp)
#print(xds_p_spec)

