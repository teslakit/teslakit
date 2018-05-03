#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# tk libs
from lib.io.matlab import ReadMatfile
from lib.predictor import CalcPCA_EstelaPred as CalcPCA

# data storage
p_data = op.join(op.dirname(__file__),'..','data')
p_test = op.join(p_data, 'tests_estela_PCA')

# use same data as matlab test to compare
p_mat_data = op.join(p_test, 'splgrd1d.mat')


# --------------------------------------
# load test data 
dmat = ReadMatfile(p_mat_data)
slp_grd_1d = dmat['SlpGrd']

# this raw data is already in [ntimeXnpoints] shape

# --------------------------------------
# PCA

# standarize predictor
print slp_grd_1d.shape
slp_grd_mean = np.mean(slp_grd_1d, axis=0)
slp_grd_std = np.std(slp_grd_1d, axis=0)
slp_grd_norm = (slp_grd_1d[:,:] - slp_grd_mean) / slp_grd_std
slp_grd_norm[np.isnan(slp_grd_norm)] = 0

# TODO: PLOTEO PARA COMPARAR DATOS MATLAB CON TKIT
xtest = xr.Dataset(
    {'test':(('time','points'),slp_grd_norm)}
)
xtest.test.plot(vmin=-1,vmax=1)
plt.show()
sys.exit()






# TODO: SEPARATE CALIBRATION AND VALIDATION USING DATE
slp_grd_norm_cal = slp_grd_norm
slp_grd_norm_val = np.array([])

# principal components analysis
from sklearn.decomposition import PCA
ipca = PCA(n_components=slp_grd_norm_cal.shape[0])
PCs = ipca.fit_transform(slp_grd_norm_cal)


print 'Principal Components Analysis COMPLETE'
xds_PCA = xr.Dataset(
    {
        'PCs': (('n_components', 'n_components'), PCs),
        'EOFs': (('n_components','n_features'), ipca.components_),
        'variance': (('n_components',), ipca.explained_variance_),
    },

    attrs = {
    }
)

print xds_PCA.PCs

# TODO STORE SAME DATA AS MATLAB CODE

