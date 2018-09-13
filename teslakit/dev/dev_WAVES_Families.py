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
from lib.objs.tkpaths import PathControl
from lib.io.matlab import ReadGowMat
from lib.waves import GetDistribution


# data storage and path control
p_data = op.join(op.dirname(__file__), '..', 'data')
pc = PathControl(p_data)


# Load wave partitions
xds_waves_parts = ReadGowMat(pc.p_st_waves_part_p1)

# calculate families
sectors = [(210, 22.5), (22.5, 135)]
xds_waves_fam = GetDistribution(xds_waves_parts, sectors)

print xds_waves_fam


# TODO: CARGAR LOS STORMS Y SEPARAR DATOS DE PARTICIONES

