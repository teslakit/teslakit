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


d1 = '1970-05-01'
d2 = '2970-05-01'
dt = 'datetime64[Y]'


dd = np.arange(d1, d2, dtype=dt)  #.astype('datetime64[D]')
print(np.diff(dd)[0])
dd = np.arange(d1 + td for td in np.diff(dd))


print(dd)
import sys; sys.exit()



date = np.datetime64('1700-06-01T00:00:00.000000000')
print(date)
print(type(date))

x = date2datenum(date)
print(x)


