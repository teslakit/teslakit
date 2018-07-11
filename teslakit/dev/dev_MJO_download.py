#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# tk libs
from lib.data_fetcher.MJO import Download_MJO

# data storage
p_data = op.join(op.dirname(__file__), '..', 'data')
p_mjo_hist = op.join(p_data, 'MJO_hist.nc')


# Download mjo and save xarray.dataset to netcdf
y1 = '1979-01-01'
xds_mjo_hist = Download_MJO(
    p_mjo_hist, init_year=y1, log=True)

