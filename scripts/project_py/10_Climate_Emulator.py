#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import ast
import pickle

# pip
import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../')

# teslakit
from teslakit.project_site import Site
from teslakit.io.matlab import ReadGowMat
from teslakit.estela import Predictor
from teslakit.climate_emulator import Climate_Emulator
from teslakit.waves import TWL_WavesFamilies, TWL_AnnualMaxima, Aggregate_WavesFamilies, Intradaily_Hydrograph


# --------------------------------------
# Site paths and parameters
data_folder = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
site = Site(data_folder, 'KWAJALEIN_TEST')

DB = site.pc.DB                        # common database
ST = site.pc.site                      # site database
PR = site.params                       # site parameters

# input files
p_wvs_parts = ST.WAVES.partitions_p1            # wave partitions (hs, tp)
p_wvs_fams_noTCs = ST.WAVES.families_notcs      # wave families (TCs removed)
p_kma_estela = op.join(ST.ESTELA.pred_slp, 'kma.nc')        # estela slp predictor KMA
p_wvs_fams_TCs_c = ST.WAVES.families_tcs_categ  # waves families at TCs time window by categories (folder)
# TODO: usar las olas correspondientes a los TCs historical radio grande

p_dwt_sim = ST.ESTELA.sim_dwt                   # daily weather types simulated with ALR

# TCs simulation input files
p_sim_r2_params = ST.TCs.sim_r2_params          # TCs parameters (copula generated) 
p_sim_r2_RBF_output = ST.TCs.sim_r2_rbf_output  # TCs numerical_IH-RBFs_interpolation output
p_probs_synth = ST.TCs.probs_synth              # synthetic TCs probabilities
p_mutau_wt = ST.ESTELA.hydrog_mutau             # intradaily WTs mu,tau data folder


# climate emulator folder
p_ce = ST.EXTREMES.climate_emulator


# --------------------------------------
# Load data for climate emulator fitting: waves partitions and families, KMA, DWT

# original wave partitions (hourly data)

xds_WVS_pts = ReadGowMat(p_wvs_parts)

# wave families (sea, swl1, swl2) without TCs
xds_WVS_fam = xr.open_dataset(p_wvs_fams_noTCs)

# ESTELA predictor KMA
xds_KMA = xr.open_dataset(p_kma_estela)

# Load DWTs sims data for climate emulator simulation
xds_DWT = xr.open_dataset(p_dwt_sim)

# Load TCs-window waves-families data by category
d_WT_TCs_wvs = {}
for k in range(6):
    p_s = op.join(p_wvs_fams_TCs_c, 'waves_fams_cat{0}.nc'.format(k))
    d_WT_TCs_wvs['{0}'.format(k)] = xr.open_dataset(p_s)



# --------------------------------------
# Load data for simulating TCs: MU, TAU

# TCs simulated with numerical and RBFs (parameters and num/RBF output)
xds_TCs_params = xr.open_dataset(p_sim_r2_params)
xds_TCs_RBFs = xr.open_dataset(p_sim_r2_RBF_output)

# Synth. TCs probabilitie changues
xds_probs_TCs = xr.open_dataset(p_probs_synth)
pchange_TCs = xds_probs_TCs['category_change_cumsum'].values[:]



# TODO: UPDATE when  to xarray.Dataset (time) mu, tau, ss
# (cambiar al generarse desde datos ESTELA)

# MU - TAU intradaily hidrographs for each WWT
l_mutau_ncs = sorted(
    [op.join(p_mutau_wt, pf) for pf in os.listdir(p_mutau_wt) if pf.endswith('.nc')]
)
xdsets_mutau_wt = [xr.open_dataset(x) for x in l_mutau_ncs]

# get only MU and TAU numpy arrays
MU_WT = np.array([x.MU.values[:] for x in xdsets_mutau_wt])
TAU_WT = np.array([x.TAU.values[:] for x in xdsets_mutau_wt])


# --------------------------------------
# MATLAB TEST DATA OVERIDE  
# TODO: delete

from teslakit.project_site import PathControl
from teslakit.io.matlab import ReadMatfile
from teslakit.custom_dateutils import datevec2datetime as d2d
from teslakit.custom_dateutils import DateConverter_Mat2Py as dmp

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


# --------------------------------------
# Climate Emulator extremes model fitting

# climate emulator object
CE = Climate_Emulator(p_ce)

# Waves and KMA bmus data share time dimension
xds_WVS_fam = xds_WVS_fam.sel(time=xds_KMA.time)
xds_WVS_pts = xds_WVS_pts.sel(time=xds_KMA.time)


# Fit extremes model
config = {
    'name_fams':       ['sea', 'swell_1', 'swell_2'],
    'force_empirical': ['sea_Tp'],
}
CE.FitExtremes(xds_KMA, xds_WVS_pts, xds_WVS_fam, config)


# Fit report figures
#CE = Climate_Emulator(p_ce)
#CE.Load()
CE.Report_Fit()


# --------------------------------------
#  Climate Emulator simulation (NO TCs)

# climate emulator object
CE = Climate_Emulator(p_ce)
CE.Load()

# Simulate waves
ls_wvs_sim = CE.Simulate_Waves(xds_DWT, d_WTTCs)


# --------------------------------------
#  Climate Emulator simulation (TCs)

# climate emulator object
CE = Climate_Emulator(p_ce)
CE.Load()

# Simulate TCs and update simulated waves
ls_tcs_sim, ls_wvs_upd = CE.Simulate_TCs(xds_DWT, d_WTTCs, xds_TCs_params, xds_TCs_RBFs, pchange_TCs, MU_WT, TAU_WT)


# --------------------------------------
#  Calculate Intradaily hydrographs for simulated storms

# iterate over simulations
for xds_wvs_sim, xds_tcs_sim in zip(ls_wvs_upd, ls_tcs_sim):

    # Aggregate waves families data
    xds_wvs_agr = Aggregate_WavesFamilies(xds_wvs_sim)

    # calculate intradaily hydrographs
    xds_hg = Intradaily_Hydrograph(xds_wvs_agr, xds_tcs_sim)

    print(xds_hg)
    print()


# --------------------------------------
# Calculate TWL annnual maxima

# iterate over simulations
for xds_wvs_sim in ls_wvs_upd:

    # Calculate TWL for waves families data
    xds_TWL = TWL_WavesFamilies(xds_wvs_sim)

    # Calculate annual maxima (manually: time index not monotonic)
    xds_TWL_AnMax = TWL_AnnualMaxima(xds_TWL)

    print(xds_TWL_AnMax)
    print()

