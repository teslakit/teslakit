#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import sys

# pip
import numpy as np
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0, op.join(op.dirname(__file__), '..', '..', '..'))

# teslakit
from teslakit.database import Database
from teslakit.climate_emulator import Climate_Emulator
from teslakit.waves import Aggregate_WavesFamilies, \
Intradaily_Hydrograph, AWL

from teslakit.plotting.waves import Plot_Waves_Histogram_FitSim
from teslakit.plotting.climate_emulator import Plot_Simulation
from teslakit.plotting.extremes import Plot_ReturnPeriodValidation



# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)
db.SetSite('KWAJALEIN')

# climate emulator
pe = db.paths.site.EXTREMES.climate_emulator
CE = Climate_Emulator(pe)
CE.Load()


# -------------------------
# load waves historical and simulated 

WVS_fit = CE.WVS_MS
WVS_sim, TCs_sim, WVS_upd = CE.LoadSim()


# Plot waves historical vs simulated histograms by family and variable
Plot_Waves_Histogram_FitSim(WVS_fit, WVS_upd)


# -------------------------
# plot annual maxima return period
nv = 'Hs'

# historical annual maxima
hist_A = CE.WVS_MS[nv].groupby('time.year').max(dim='time')
print(hist_A)
print()

# simulation annual maxima 
sim_A = SIM_WAVES_h[nv].groupby('time.year').max(dim='time')
print(sim_A)


# TODO: test fast
sim_A.to_netcdf('tenp_sim_A.nc')
hist_A.to_netcdf('tenp_hist_A.nc')

# Plot Return Period graph
Plot_ReturnPeriodValidation(hist_A, sim_A)


