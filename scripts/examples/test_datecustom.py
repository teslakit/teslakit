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
sys.path.insert(0, op.join(op.dirname(__file__), '..', '..'))

# teslakit
from teslakit.util.time_operations import date2datenum


# --------------------------------------
# test date2datenum 

date = np.datetime64('1700-06-01T00:00:00.000000000')
print(date)
print(type(date))

x = date2datenum(date)
print(x)


