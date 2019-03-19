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
from lib.io.matlab import ReadMatfile as rmf
from lib.extremes import Climate_Emulator
from lib.custom_dateutils import DateConverter_Mat2Py as d2d


# --------------------------------------
# files
p_data = op.join(op.dirname(__file__),'..','data')
p_data = op.join(p_data, 'tests', 'test_ExtremesGEV')


# --------------------------------------
# Load test data (original matlab files)

# waves and KMA MaxStorm
xds_WT_wvs = xr.open_dataset(op.join(p_data, 'xds_WT_wvs_fam_noTCs.nc'))
xds_WT_KMA = xr.open_dataset(op.join(p_data, 'xds_WT_KMA.nc'))

# DWTs (Daily Weather Types simulated using ALR)
p_DWTs = op.join(p_data, 'Nico_Montecarlo', 'DWT_1000years_mjo_awt_v2.mat')
dm_DWTs = rmf(p_DWTs)
daily_WT = dm_DWTs['bmusim'].T
dates_DWT = dm_DWTs['datesim']  # TODO: convertir a python


# TCs simulated with numerical and RBFs (TCs params, and TCs num/RBF output)
p_TCs = op.join(
    p_data, 'Nico_Montecarlo', 'Reconstructed_waves_MULTIVARIATE_Pmean.mat'
)
dm_TCs = rmf(p_TCs)

TCs_params = dm_TCs['Param']  # TCs parameters: pressure_mean, pressure_min...
TCs_output = dm_TCs['Results']  # TCs numerical output:  Hs, Tp, Dir, mu, SS

xds_TCs = xr.Dataset(
    {
        'pressure_min':(('storm'), TCs_params[:,0]),
        'hs':(('storm'), TCs_output[:,0]),
        'tp':(('storm'), TCs_output[:,1]),
        'dir':(('storm'), TCs_output[:,4]),
        'mu':(('storm'), TCs_output[:,5]),
        'ss':(('storm'), TCs_output[:,2]),
    },
    coords = {
        'storm':(('storm'), np.arange(TCs_params.shape[0]))
    },
)

# MU TAU intradaily hydrographs for each WT (36)
p_MUTAU = op.join(p_data, 'Nico_Montecarlo', 'MU_TAU_36WT.mat')
dm_MUTAU = rmf(p_MUTAU)
MU_WT = dm_MUTAU['MU_WT']
TAU_WT = dm_MUTAU['TAU_WT']

# TCs probs change
p_probsTCs = op.join(p_data, 'Nico_Montecarlo', 'Probabilities_Historical.mat')
pchange_TCs = rmf(p_probsTCs)['p_sint']




# TODO: SEGUIR AQUI

# DAILY WTS SIM
# WT40, WT41, WT42...
# parametros extremes: PARAM_GEV_SAMPLE, PROB_CHROM, SIGMA, 




# --------------------------------------
# test chromosomes

# variables to get chrom probs
