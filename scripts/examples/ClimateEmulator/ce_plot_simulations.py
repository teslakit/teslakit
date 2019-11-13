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
from teslakit.waves import AnnualMaxima

from teslakit.plotting.waves import Plot_Waves_Histogram_FitSim
from teslakit.plotting.output import Plot_Output
from teslakit.plotting.extremes import Plot_ReturnPeriodValidation



# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)
db.SetSite('KWAJALEIN')

# climate emulator
CE = Climate_Emulator(db.paths.site.EXTREMES.climate_emulator)
CE.Load()

# test path
p_test = op.join(p_data, 'tests', 'ClimateEmulator', 'plot_output')


# -------------------------
# load waves famileis fit-sim 
#xds_wvs_fit = xr.open_dataset(op.join(p_test, 'waves_fit.nc'))
#xds_wvs_sim = xr.open_dataset(op.join(p_test, 'waves_sim.nc'))

#Plot_Waves_Histogram_FitSim(xds_wvs_fit, xds_wvs_sim)


# -------------------------
# load climate emulator full output 
SIM = xr.open_dataset(op.join(p_test, 'waves_output.nc'))

#Plot_Output(SIM)


# -------------------------
# plot annual maxima return period
nv = 'AWL'

# historical annual maxima
hist_A = AnnualMaxima(CE.WVS_MS, nv)

# load climate emulator annual maxima
sim_A = AnnualMaxima(SIM, nv)

# Plot Return Period graph
Plot_ReturnPeriodValidation(hist_A, sim_A, nv)


