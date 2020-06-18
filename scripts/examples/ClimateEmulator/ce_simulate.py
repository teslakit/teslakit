#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op
import sys

# DEV: override installed teslakit
import sys
sys.path.insert(0, op.join(op.dirname(__file__), '..', '..', '..'))

# pip 
import numpy as np
import xarray as xr

# teslakit
from teslakit.database import Database
from teslakit.climate_emulator import Climate_Emulator


# --------------------------------------
# Test data storage

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
p_test = op.join(p_data, 'tests', 'ClimateEmulator', 'CE_FitExtremes')


# --------------------------------------
# Test 1 - 3 waves families, chromosomes on

p_t = op.join(p_test, 'test_1')

# DWTs to simulate
p_DWTs_sim = op.join(p_t, 'DWT_sim.nc')

DWTs_sim = xr.open_dataset(p_DWTs_sim)
DWTs_sim = DWTs_sim.isel(time = slice(0, 365*5), n_sim=0)


# Climate Emulator object 
p_ce = op.join(p_t, 'ce')
CE = Climate_Emulator(p_ce)
CE.Load()


# Simulate Max. Storms Waves (No TCs)
WVS_sim = CE.Simulate_Waves(DWTs_sim)
print(WVS_sim)
print()


# Load data (needed to simulate WITH TCs)

#TCs_params = db.Load_TCs_r2_sim_params()    # TCs parameters (copula generated)
#TCs_RBFs = db.Load_TCs_sim_r2_rbf_output()  # TCs numerical_IH-RBFs_interpolation output

#probs_TCs =  db.Load_TCs_probs_synth()      # TCs synthetic probabilities
#pchange_TCs = probs_TCs['category_change_cumsum'].values[:]

#l_mutau_wt = db.Load_MU_TAU_hydrograms()   # MU - TAU intradaily hidrographs for each WWT
#MU_WT = np.array([x.MU.values[:] for x in l_mutau_wt])  # MU and TAU numpy arrays
#TAU_WT = np.array([x.TAU.values[:] for x in l_mutau_wt])


# Simulate Max. Storms Waves (No TCs)

#TCs_sim, WVS_upd = CE.Simulate_TCs(
#    DWTs_sim, TCs_params, TCs_RBFs, pchange_TCs, MU_WT, TAU_WT
#)
#print(WVS_upd)
#print()
#print(TCs_sim)


# --------------------------------------
# Test 2 - MAJURO - 4 waves families, chromosomes off, extra variables

p_t = op.join(p_test, 'test_2')


# DWTs to simulate
#p_DWTs_sim = op.join(p_t, 'DWT_sim.nc')

#DWTs_sim = xr.open_dataset(p_DWTs_sim)
#DWTs_sim = DWTs_sim.isel(time = slice(0, 365*5), n_sim=0)

# TODO: usando los del test_1
aux = DWTs_sim.evbmus_sims.values[:]
aux[aux>35] = 35
DWTs_sim['evbmus_sims'] = (('time',), aux)


# Climate Emulator object 
p_ce = op.join(p_t, 'ce')
CE = Climate_Emulator(p_ce)
CE.Load()


# Simulate Max. Storms Waves (No TCs)
WVS_DATA_sim = CE.Simulate_Waves(DWTs_sim)
print(WVS_DATA_sim)
print()

