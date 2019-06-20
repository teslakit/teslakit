#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..','..'))

# python libs
import numpy as np
import xarray as xr

# custom libs
from teslakit.project_site import PathControl, Site
from teslakit.climate_emulator import Climate_Emulator
from teslakit.custom_dateutils import datevec2datetime as d2d
from teslakit.custom_dateutils import DateConverter_Mat2Py as dmp
from teslakit.io.matlab import ReadMatfile


# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test_ce = op.join(p_tests, 'ClimateEmulator', 'CE_FitExtremes')

# input
p_ce = op.join(p_test_ce, 'ce')  # climate emulator



# --------------------------------------
#  MATLAB TEST DATA    


# Test data storage
pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ClimateEmulator', 'ml_jupyter')


# load test KMA (bmus, time, number of clusters, cenEOFs)
p_bmus = op.join(p_test, 'bmus_testearpython.mat')
dmatf = ReadMatfile(p_bmus)
xds_KMA = xr.Dataset(
    {
        'bmus'       : ('time', dmatf['KMA']['bmus']),
        'cenEOFs'    : (('n_clusters', 'n_features',), dmatf['KMA']['cenEOFs']),
    },
    coords = {'time' : np.array(d2d(dmatf['KMA']['Dates']))}
)

# DWTs (Daily Weather Types simulated using ALR)
p_DWTs = op.join(p_test, 'DWT_1000years_mjo_awt_v2.mat')
dm_DWTs = ReadMatfile(p_DWTs)
xds_DWT = xr.Dataset(
    {
        'evbmus_sims' : (('time', 'n_sim'), dm_DWTs['bmusim'].T),
    },
    coords = {'time' : dmp(dm_DWTs['datesim'])}
)

# get WTs37, 42 from matlab file
p_WTTCs = op.join(p_test, 'KWA_waves_2PART_TCs_nan.mat')
dm_WTTCs = ReadMatfile(p_WTTCs)

# Load TCs-window waves-families data by category
d_WTTCs = {}
for i in range(6):

    k = 'wt{0}'.format(i+1+36)
    sd = dm_WTTCs[k]

    d_WTTCs['{0}'.format(i+1+36)] = xr.Dataset(
        {
            'sea_Hs'      : (('time',), sd['seaHs']),
            'sea_Dir'     : (('time',), sd['seaDir']),
            'sea_Tp'      : (('time',), sd['seaTp']),
            'swell_1_Hs'  : (('time',), sd['swl1Hs']),
            'swell_1_Dir' : (('time',), sd['swl1Dir']),
            'swell_1_Tp'  : (('time',), sd['swl1Tp']),
            'swell_2_Hs'  : (('time',), sd['swl2Hs']),
            'swell_2_Dir' : (('time',), sd['swl2Dir']),
            'swell_2_Tp'  : (('time',), sd['swl2Tp']),
        }
    )



# TODO: for testing
xds_DWT = xds_DWT.isel(time=slice(0,500), n_sim=slice(0,1))


# --------------------------------------
# Climate Emulator object 
CE = Climate_Emulator(p_ce)
CE.Load()



# --------------------------------------
# Simulate Max. Storms Waves (No TCs)
ls_wvs_sim = CE.Simulate_Waves(xds_DWT, d_WTTCs)
print(ls_wvs_sim[0])
print()






# --------------------------------------
# Load data (needed to simulate WITH TCs)

p_input_TCs = \
'/Users/nico/Projects/TESLA-kit/source/data/sites/KWAJALEIN_TEST/'

p_sim_r2_params = op.join(p_input_TCs, 'TCs', 'TCs_sim_r2_params.nc')
p_sim_r2_RBF_output = op.join(p_input_TCs, 'TCs', 'TCs_sim_r2_RBF_output.nc')
p_probs_synth = op.join(p_input_TCs, 'TCs', 'TCs_synth_ProbsChange.nc')
p_mutau_wt = op.join(p_input_TCs, 'ESTELA', 'hydrographs')


# TCs simulated with numerical and RBFs (parameters and num/RBF output)
xds_TCs_params = xr.open_dataset(p_sim_r2_params)
xds_TCs_RBFs = xr.open_dataset(p_sim_r2_RBF_output)

# Synth. TCs probabilitie changues
xds_probs_TCs = xr.open_dataset(p_probs_synth)
pchange_TCs = xds_probs_TCs['category_change_cumsum'].values[:]

# MU - TAU intradaily hidrographs for each WWT
l_mutau_ncs = sorted(
    [op.join(p_mutau_wt, pf) for pf in os.listdir(p_mutau_wt) if pf.endswith('.nc')]
)
xdsets_mutau_wt = [xr.open_dataset(x) for x in l_mutau_ncs]

# get only MU and TAU numpy arrays
MU_WT = np.array([x.MU.values[:] for x in xdsets_mutau_wt])
TAU_WT = np.array([x.TAU.values[:] for x in xdsets_mutau_wt])




# --------------------------------------
# Simulate Max. Storms Waves (No TCs)
ls_TCs_sims, ls_wvs_sims_upd = CE.Simulate_TCs(
    xds_DWT, d_WTTCs, xds_TCs_params, xds_TCs_RBFs, pchange_TCs, MU_WT, TAU_WT
)
print()
print(ls_wvs_sims_upd[0])
print()
print(ls_TCs_sims[0])
print('test done.')

