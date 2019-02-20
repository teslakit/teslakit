#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import xarray as xr
import numpy as np

# custom libs
from lib.io.matlab import ReadMatfile
from lib.RBF import RBF_Reconstruction


# --------------------------------------
# files
p_data = op.join(op.dirname(__file__),'..','data')
p_test = op.join(p_data,'tests','tests_RBF')

# input
p_subset = op.join(p_test, 'MDA_1000.mat')
p_target = op.join(p_test, 'out_subset_norm.mat')
p_dataset = op.join(p_test, 'MULTIVARIATE_100000parameters.mat')

# output
p_out = op.join(p_test, 'output_dataset_norm.npy')


# load subset-target data
subset = ReadMatfile(p_subset)['Subset']  # pmean, vmean, gamma, delta
target = ReadMatfile(p_target)['datos_n']  # Hs, Tp, SS, TWL, Dir, MU

# load full dataset for interpolation
dm = ReadMatfile(p_dataset)
dataset= np.column_stack(
    (dm['PMEAN1'], dm['VMEAN'], dm['GAMMA'], dm['DELTA'])
)

# subset will be normalized inside RBF_reconstruction
# target and dataset data are previously normalized but it is not a requirement


# --------------------------------------
# RBF

# subset - scalar / directional indexes
ix_scalar_subset = [0,1]        # scalar (pmean,  vmean)
ix_directional_subset = [2,3]   # directional (delta, gamma)

# target - scalar / directional indexes
ix_scalar_target = [0,1,2,3,5]  # scalar (Hs, Tp, SS, TWL, MU)
ix_directional_target = [4]     # directional (Dir)

output = RBF_Reconstruction(
    subset, ix_scalar_subset, ix_directional_subset,
    target, ix_scalar_target, ix_directional_target,
    dataset)

print(output)
np.save(p_out, output)

