#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op

# pip 
import numpy as np
import xarray as xr

# DEV: override installed teslakit
import sys
sys.path.insert(0, op.join(op.dirname(__file__), '..', '..', '..'))

# teslakit
from teslakit.climate_emulator import Climate_Emulator
from teslakit.waves import AWL


# --------------------------------------
# Test data storage

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
p_test = op.join(p_data, 'tests', 'ClimateEmulator', 'CE_FitExtremes')


# --------------------------------------
# Test 1 - ROI - 3 waves families, chromosomes on

p_t = op.join(p_test, 'test_1')

# input kma and waves
p_KMA = op.join(p_t, 'kma.nc')
p_WVS = op.join(p_t, 'waves_historical.nc')


# Load and prepare data 
KMA = xr.open_dataset(p_KMA)
WVS = xr.open_dataset(p_WVS)

DWTs_fit = xr.Dataset(
    {
        'bmus'       : ('time', KMA['sorted_bmus_storms'].values[:] + 1),
        'cenEOFs'    : (('n_clusters', 'n_features',), KMA['cenEOFs'].values[:]),
    },
    coords = {'time' : KMA['time'].values[:]}
)
WVS_fit = WVS.sel(time = slice(DWTs_fit.time[0], DWTs_fit.time[-1]))

# calculate proxy variable: AWL
WVS_fit['AWL'] = AWL(WVS_fit['Hs'], WVs_fit['Tp'])

print(DWTs_fit)
print()
print(WVS_fit)
print()


# Climate Emulator object 
p_ce = op.join(p_t, 'ce')
CE = Climate_Emulator(p_ce)


# WVS_fit contains a set of Hs, Tp, Dir for each waves_family (fam_Hs, fam_Tp, fam_Dir)
# default distributions (modify at config['distribution'] dictionary)
# Hs, Tp          - GEV
# Dir             - Empirical

config = {
    'waves_families': ['sea', 'swell_1', 'swell_2'],
    'distribution': [
        ('sea_Tp', 'Empirical'),
    ],
    'do_chromosomes': True,
}

# Fit climate emulator and save 
CE.FitExtremes(DWTs_fit, WVS_fit, config, proxy='AWL')

print('test 1 done.')
print()


# --------------------------------------
# Test 2 - MAJURO - 4 waves families, chromosomes off, extra variables

p_t = op.join(p_test, 'test_2')

# input kma and waves
p_KMA = op.join(p_t, 'kma.nc')
p_WVS = op.join(p_t, 'waves_historical.nc')

# extra variables
p_WDS = op.join(p_t, 'Winds.nc')
p_PSS = op.join(p_t, 'Pressures.nc')


# Load and prepare data 
KMA = xr.open_dataset(p_KMA)
WVS = xr.open_dataset(p_WVS)
WDS = xr.open_dataset(p_WDS)
PSS = xr.open_dataset(p_PSS)

DWTs_fit = xr.Dataset(
    {
        'bmus'       : ('time', KMA['sorted_bmus'].values[:] + 1),
        'cenEOFs'    : (('n_clusters', 'n_features',), KMA['cenEOFs'].values[:]),
    },
    coords = {'time' : KMA['time'].values[:]}
)
WVS_fit = WVS.sel(time = slice(DWTs_fit.time[0], DWTs_fit.time[-1]))

# calculate proxy variable: AWL
WVS_fit['AWL'] = AWL(WVS_fit['Hs'], WVs_fit['Tp'])

# resample extra data
WDS = WDS.resample(time='1h').pad().sel(time=WVS_fit.time)
PSS = PSS.resample(time='1h').pad().sel(time=WVS_fit.time)

# add extra data to WAVES dataset
WVS_fit['wind_speed'] = WDS.Speed
WVS_fit['wind_dir'] = WDS.Dir
WVS_fit['msl'] = PSS.msl

print(DWTs_fit)
print()
print(WVS_fit)
print()


# Climate Emulator object 
p_ce = op.join(p_t, 'ce')
CE = Climate_Emulator(p_ce)


# WVS_fit contains a set of Hs, Tp, Dir for each waves_family (fam_Hs, fam_Tp, fam_Dir)
# WVS_fit also contains all optional extra variables (independent of waves families) 

# default distributions (use config['distribution'] to change)
# Hs, Tp          - GEV
# Dir             - Empirical
# extra_variables - GEV

config = {
    'waves_families': ['sea', 'swell_1', 'swell_2', 'swell_3'],
    'extra_variables': ['wind_speed', 'wind_dir', 'msl'],
    'distribution': [
        ('sea_Tp', 'Empirical'),
        ('wind_dir', 'Empirical'),
        #('wind_speed', 'Weibull'),
    ],
    'do_chromosomes': False,
}

# Fit climate emulator and save 
CE.FitExtremes(DWTs_fit, WVS_fit, config, proxy = 'AWL')

print('test 2 done.')
print()

