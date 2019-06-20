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
from teslakit.project_site import PathControl
from teslakit.climate_emulator import Climate_Emulator
from teslakit.waves import Aggregate_WavesFamilies, Intradaily_Hydrograph


# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ClimateEmulator', 'CE_FitExtremes')

# input
p_ce = op.join(p_test, 'ce')


# --------------------------------------
# Climate Emulator object 
CE = Climate_Emulator(p_ce)

# load previously simulated storms (without TCs)
ls_wvs_upd, ls_tcs_sim = CE.LoadSim(TCs=True)

# iterate over simulations
for xds_wvs_sim, xds_tcs_sim in zip(ls_wvs_upd, ls_tcs_sim):

    # Aggregate waves families data 
    xds_wvs_agr = Aggregate_WavesFamilies(xds_wvs_sim)

    # calculate intradaily hydrographs
    xds_hg = Intradaily_Hydrograph(xds_wvs_agr, xds_tcs_sim)

    print(xds_hg)
    print()
