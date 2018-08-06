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

# TODO: VAMOS A COMPROBAR LOS RESULTADOS FRENTE A MATLAB Y SEGUIMOS

p_temp = op.join(p_waves, 'test')
for var in ['Hs','Tp','Dir']:
    np.savetxt(
        op.join(p_temp,'sea_{0}.txt'.format(var)),
        xds_waves_fam['sea_{0}'.format(var)].values[:]
    )
    np.savetxt(
        op.join(p_temp,'swell_1_{0}.txt'.format(var)),
        xds_waves_fam['swell_1_{0}'.format(var)].values[:]
    )
    np.savetxt(
        op.join(p_temp,'swell_2_{0}.txt'.format(var)),
        xds_waves_fam['swell_2_{0}'.format(var)].values[:]
    )


# TODO: USAR EL ARCHIVO DE HURACANES PARA SEPARAR WAVES-CYCLONES

