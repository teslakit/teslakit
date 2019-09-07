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
from teslakit.alr import ALR_WRP


# --------------------------------------
# Test data storage

p_tests = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data/tests'

# MJO ALR wrap
p_mjo_alrw = op.join(p_tests, 'MJO', 'alr_w')
ALRW = ALR_WRP(p_mjo_alrw)

# show model report 
ALRW.Report_Fit()

# show sim report
ALRW.Report_Sim()



