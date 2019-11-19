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
from teslakit.util.time_operations import datevec2datetime as d2d
from teslakit.util.time_operations import DateConverter_Mat2Py as dmp
from teslakit.io.matlab import ReadMatfile


# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')


# DWTs simulation
DWTs_sim = db.Load_ESTELA_DWT_sim()

# DWTs simulation and time
DWTs_sim = DWTs_sim.isel(time=slice(0,365*5), n_sim=0)

# --------------------------------------
# Climate Emulator object 
CE = Climate_Emulator(db.paths.site.EXTREMES.climate_emulator)
CE.Load()


# --------------------------------------
# Simulate Max. Storms Waves (No TCs)
WVS_sim = CE.Simulate_Waves(DWTs_sim, n_sims=5)
print(WVS_sim)


# --------------------------------------
# Load data (needed to simulate WITH TCs)

TCs_params = db.Load_TCs_r2_sim_params()    # TCs parameters (copula generated)
TCs_RBFs = db.Load_TCs_sim_r2_rbf_output()  # TCs numerical_IH-RBFs_interpolation output

probs_TCs =  db.Load_TCs_probs_synth()      # TCs synthetic probabilities
pchange_TCs = probs_TCs['category_change_cumsum'].values[:]

l_mutau_wt = db.Load_MU_TAU_hydrograms()   # MU - TAU intradaily hidrographs for each WWT
MU_WT = np.array([x.MU.values[:] for x in l_mutau_wt])  # MU and TAU numpy arrays
TAU_WT = np.array([x.TAU.values[:] for x in l_mutau_wt])


# --------------------------------------
# Simulate Max. Storms Waves (No TCs)

TCs_sim, WVS_upd = CE.Simulate_TCs(
    DWTs_sim, TCs_params, TCs_RBFs, pchange_TCs, MU_WT, TAU_WT
)
print(WVS_upd)
print()
print(TCs_sim)

