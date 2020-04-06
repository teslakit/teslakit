#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op

# python libs
import numpy as np
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0, op.join(op.dirname(__file__), '..', '..', '..'))

# teslakit
from teslakit.database import Database
from teslakit.climate_emulator import Climate_Emulator
from teslakit.waves import Aggregate_WavesFamilies, Intradaily_Hydrograph


# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')


# --------------------------------------
# Climate Emulator object 
CE = Climate_Emulator(db.paths.site.EXTREMES.climate_emulator)
CE.Load()

# load previously simulated storms (without TCs)
WVS_sim, TCs_sim, WVS_upd = CE.LoadSim()

# cut data
#WVS_upd = WVS_upd.sel(time=slice('1700-01-01','1710-01-01'))
#TCs_sim = TCs_sim.sel(time=slice('1700-01-01','1710-01-01'))

# Aggregate waves families data 
wvs_agr = Aggregate_WavesFamilies(WVS_upd.sel(n_sim=0))

# calculate intradaily hydrographs
hy = Intradaily_Hydrograph(wvs_agr, TCs_sim.sel(n_sim=0))

print(hy)
print()

