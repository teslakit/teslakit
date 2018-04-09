#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import xarray as xr

# tk libs
from lib.custom_stats import sort_cluster_gen_corr_end

# data storage
p_data = op.join(op.dirname(__file__),'..','data')

# sort kmeans
dt = xr.open_dataset(op.join(p_data,'test_sortcgce.nc'))
print sort_cluster_gen_corr_end(dt['kma_cc'].values, 6)


