#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xarray as xr
import os.path as op

from lib.custom_stats import sort_cluster_gen_corr_end

# data storage
p_data = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/'

# TODO: REPROGRAMAR LA  UNCION DESDE MATLAB
dt = xr.open_dataset(op.join(p_data,'test_sortcgce.nc'))
print sort_cluster_gen_corr_end(dt['kma_cc'].values, 6)


