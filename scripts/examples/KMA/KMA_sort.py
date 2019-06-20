#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..','..'))

# pip
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../../')

# teslakit
from teslakit.kma import sort_cluster_gen_corr_end

# TODO: revisar datos test

# --------------------------------------
# test data storage

pc = pathcontrol()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'estela', 'test_estela_pca')


# data storage
p_data = op.join(op.dirname(__file__),'..','data')

# sort kmeans
dt = xr.open_dataset(op.join(p_data,'test_sortcgce.nc'))
print(sort_cluster_gen_corr_end(dt['kma_cc'].values, 6))

