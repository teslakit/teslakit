#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op

# python libs
import numpy as np
from scipy.stats import  gumbel_l, genextreme

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../../')

# teslakit
from teslakit.io.matlab import ReadMatfile as rmf
from teslakit.project_site import PathControl
from teslakit.extremes import ACOV


# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ClimateEmulator', 'gev_acov')

# input files
p_mat = op.join(p_test, 'gevacov.mat')


# --------------------------------------
# Load GEV params and data from matlab file 
dm = rmf(p_mat)['mldata']
shape, scale, loc = dm['param_GEV']
data = dm['data']

shape = shape*-1  # matlab and python use different shape

# nlogL value
nLogL = genextreme.nnlf((shape, loc, scale), data)

print(shape, loc, scale)
print(nLogL)
print()

# asynptotic variances
theta = (shape, loc, scale)
acov = ACOV(genextreme.nnlf, theta, data)
print(acov)

