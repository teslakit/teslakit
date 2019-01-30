#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import xarray as xr
import numpy as np

# tk libs
from lib.objs.tkpaths import Site
from lib.data_fetcher.STORMS import Download_NOAA_WMO
from lib.tcyclone import Extract_Circle
from lib.plotting.storms import WorldMap_Storms


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')

DB = site.pc.DB                        # common database
ST = site.pc.site                      # site database
PR = site.params                       # site parameters

# output files
p_hist_tcs = DB.TCs.noaa


# --------------------------------------
# Download TCs and save xarray.dataset to netcdf

xds_wmo = Download_NOAA_WMO(p_hist_tcs)
print('\nHistorical TCs stored at:\n{0}'.format(p_hist_tcs))

