#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# tk libs
from lib.objs.tkpaths import PathControl
from lib.io.cfs import ReadSLP


# --------------------------------------
# data storage and path control
pc = PathControl()
pc.SetSite('test_site')


# --------------------------------------
# site coordinates 
lat1 = 60.5
lat2 = 0.5
lon1 = 115
lon2 = 279
resample = 4  #2º

# load predictor data (SLP) from CFSR and save to .nc 
xds_SLP_site = ReadSLP(
    pc.p_db_slp,
    lat1, lat2, lon1, lon2, resample,
    p_save=pc.site.est.slp)

