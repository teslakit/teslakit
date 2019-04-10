#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..','..'))

# tk libs
from teslakit.project_site import Site
from teslakit_downloader.MJO import Download_MJO


# --------------------------------------
# test data storage

site = Site('KWAJALEIN_TEST')
DB = site.pc.DB                        # common database
p_mjo_hist = DB.MJO.hist               # historical MJO

# Download MJO and save to netcdf
y1 = '1979-01-01'
xds_mjo_hist = Download_MJO(DB.MJO.hist, init_year=y1, log=True)
print(xds_mjo_hist)

