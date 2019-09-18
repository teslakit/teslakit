#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import sys

# pip
import numpy as np
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0, op.join(op.dirname(__file__), '..', '..', '..'))

# teslakit
from teslakit.database import Database
from teslakit.alr import ALR_WRP


# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')

# alr test
p_alrw = db.paths.site.SST.alrw
#p_alrw = db.paths.site.MJO.alrw
#p_alrw = db.paths.site.ESTELA.alrw

# ALR wrap
ALRW = ALR_WRP(p_alrw)

# show model report 
#ALRW.Report_Fit(terms_fit=True, summary=True, export=True)

# show simulation report
ALRW.Report_Sim(py_month_ini=6)

