#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# tk libs
from lib.objs.tkpaths import PathControl
from lib.data_fetcher.MJO import Download_MJO


# --------------------------------------
# data storage and path control
pc = PathControl()
pc.SetSite('test_site')


# Download MJO and save to netcdf
y1 = '1979-01-01'
xds_mjo_hist = Download_MJO(
    pc.DB.mjo.hist, init_year=y1, log=True)

