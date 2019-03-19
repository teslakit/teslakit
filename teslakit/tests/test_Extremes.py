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
from lib.extremes import Correlation_Smooth_Partitions as CorrSP


# --------------------------------------
# files
p_data = op.join(op.dirname(__file__),'..','data')
p_data = op.join(p_data, 'tests', 'test_ExtremesGEV')




# --------------------------------------
# Load test data
xds_WT_wvs = xr.open_dataset(op.join(p_data, 'xds_WT_wvs_fam_noTCs.nc'))
xds_WT_KMA = xr.open_dataset(op.join(p_data, 'xds_WT_KMA.nc'))

# data to use
n_clusters = 36
bmus = xds_WT_KMA.bmus.values
cenEOFs = xds_WT_KMA.cenEOFs.values


# --------------------------------------
# test chromosomes

# variables to get chrom probs
#se_Hs = xds_WT_wvs.sea_Hs.values
#s1_Hs = xds_WT_wvs.swell_1_Hs.values
#s2_Hs = xds_WT_wvs.swell_2_Hs.values
#np_cvs = np.column_stack([se_Hs, s1_Hs, s2_Hs])

#chrom, chrom_probs = ChromProbs(bmus, n_clusters, np_cvs)


# --------------------------------------
# Sigma, correlacion Tm, Hs suavizado partitions 

# load stored data
#xds_gev_params = xr.open_dataset(op.join(p_data, 'test_xds_gev_paramsfit.nc'))
#chrom = np.load(op.join(p_data, 'chrom.npy'))

# TODO: repasar args y documentar 
# get sigma
#wvs_fams = ['sea', 'swell_1', 'swell_2']
#sigma = CorrSP(
#    bmus, cenEOFs, n_clusters, xds_WT_wvs, wvs_fams,
#    xds_gev_params, chrom)


# --------------------------------------
# test GEV: Fit 

# test one waves var
#var = xds_WT_wvs.swell_1_Hs.values

# Fit
#gev_params = FitGEV(bmus, n_clusters, var)
#print(gev_params)
#print()


# --------------------------------------
# test GEV: Sample and climate emulator
xds_gev_params = xr.open_dataset(op.join(p_data, 'test_xds_gev_paramsfit.nc'))
chrom = np.load(op.join(p_data, 'chrom.npy'))

# TODO: Sample GEV parameters for simulation
#paramS = SampleGEV(bmus, n_clusters, gev_params, var)
print(paramS)
print()


