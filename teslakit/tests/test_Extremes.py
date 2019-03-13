
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
from lib.extremes import FitGEV_KMA_Frechet as FitGEV
from lib.extremes import SampleGEV_KMA_Smooth as SampleGEV

from lib.extremes import ChromosomesProbabilities_KMA as ChromProbs


# --------------------------------------
# files
p_data = op.join(op.dirname(__file__),'..','data')
p_data = op.join(p_data, 'tests', 'test_ExtremesGEV')

p_WT_wvs = op.join(p_data, 'xds_WT_wvs.nc')
p_WT_KMA = op.join(p_data, 'xds_WT_KMA.nc')

xds_WT_wvs = xr.open_dataset(p_WT_wvs)
xds_WT_KMA = xr.open_dataset(p_WT_KMA)


# --------------------------------------
# test chromosomes
bmus = xds_WT_KMA.bmus.values
n_clusters = 36
var = xds_WT_wvs.swell_1_Hs.values

# variables to get chrom probs
se_Hs = xds_WT_wvs.sea_Hs.values
s1_Hs = xds_WT_wvs.swell_1_Hs.values
s2_Hs = xds_WT_wvs.swell_2_Hs.values
np_cvs = np.column_stack([se_Hs, s1_Hs, s2_Hs])

chrom, chrom_probs = ChromProbs(bmus, n_clusters, np_cvs)
print(chrom)
print()
print(chrom_probs)
print()


# --------------------------------------
# TODO Sigma, correlacion Tm,Hs suavizado partitions 


# --------------------------------------
# test GEV: Fit and Sample
bmus = xds_WT_KMA.bmus.values
n_clusters = 36
var = xds_WT_wvs.swell_1_Hs.values

# Fit
gev_params = FitGEV(bmus, n_clusters, var)
print(gev_params)
print()

# TODO: Sample GEV parameters for simulation
paramS = SampleGEV(bmus, n_clusters, gev_params, var)
print(paramS)
print()

