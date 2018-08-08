#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import numpy as np
import xarray as xr

# tk libs
from lib.io.matlab import ReadGowMat
from lib.waves import GetDistribution

# data storage
p_data = op.join(op.dirname(__file__), '..', 'data')
p_waves = op.join(p_data, 'WAVES')

# input data
p_waves_parts_p1 = op.join(p_waves, 'partitions', 'point1.mat' )


# Load wave partitions
xds_waves_parts = ReadGowMat(p_waves_parts_p1)

# calculate families
sectors = [(210, 22.5), (22.5, 135)]
xds_waves_fam = GetDistribution(xds_waves_parts, sectors)
print xds_waves_fam
print ''

