#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op

# pip
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import h5py

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../../')

# teslakit
from teslakit.project_site import PathControl
from teslakit.io.matlab import ReadMatfile as rmat
from teslakit.custom_dateutils import datevec2datetime
from teslakit.plotting.wts import Plot_PerpYear, Plot_Compare_PerpYear
from teslakit.plotting.ALR import Plot_Covariate, Plot_Compare_Covariate
from teslakit.custom_dateutils import xds2datetime as x2d


# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ALR')

p_tests_plot_data = op.join(p_test, 'tests_plot', 'data')


# bmus and dates sim (dates previously fixed)
p_kma_sim = op.join(p_tests_plot_data,'TAIRUA_v1.mat')

# bmus and dates historical (dates previously fixed)
p_kma_hist = op.join(p_tests_plot_data,'DWT_NZ_16_fixed.mat')

# MJO and PCs
p_mjo = op.join(p_tests_plot_data,'MJO_june.mat')
p_pcs = op.join(p_tests_plot_data,'PCs_for_AWT.mat')


# READ data (to xr.dataset)
d_mat = rmat(p_kma_sim)
xds_KMA_sims = xr.Dataset(
    {
        'bmus':(('time','n_sim'), np.transpose(d_mat['bmusim'])),
    },
    coords = {
        'time': [datetime(r[0],r[1],r[2]) for r in d_mat['datesim']],
    }
)

d_mat = rmat(p_kma_hist)
xds_KMA_hist = xr.Dataset(
    {
        'bmus':(('time',), d_mat['bmus']),
    },
    coords = {
        'time': [datetime(r[0],r[1],r[2]) for r in d_mat['dates']],
    }
)

d_mat = rmat(p_pcs)['AWT']
xds_PCs = xr.Dataset(
    {
        'PC1':(('time',), d_mat['PCs'][:,0]),
        'PC2':(('time',), d_mat['PCs'][:,1]),
        'PC3':(('time',), d_mat['PCs'][:,2]),
    },
    coords = {
        'time': [datetime(r[0],r[1],r[2]) for r in d_mat['Dates']],
    }
)




# ---------------------------------------------------
# TEST: compare perpetual year plot
num_wts = 16
num_sim = 1000

# hist
time_hist = [x2d(t) for t in xds_KMA_hist.time]
bmus_values_hist = np.reshape(xds_KMA_hist.bmus.values,[-1,1])
Plot_PerpYear(bmus_values_hist, time_hist, num_wts)

# sim
#num_sim = 1000
time_sim = [x2d(t) for t in xds_KMA_sims.time]
bmus_values_sim = xds_KMA_sims.bmus.values
Plot_PerpYear(bmus_values_sim, time_sim, num_wts, num_sim)

# compare
Plot_Compare_PerpYear(
    num_wts,
    bmus_values_sim, time_sim,
    bmus_values_hist, time_hist,
    n_sim = num_sim
)


# ---------------------------------------------------
# TEST: compare PCS

# Plot PC1 - historical
xds_PCs_hist = xds_PCs.sel(
    time=slice(time_hist[0],time_hist[-1])
)
time_hist_covars = [x2d(t) for t in xds_PCs_hist.time]
Plot_Covariate(
    bmus_values_hist, xds_PCs_hist.PC1.values,
    time_hist, time_hist_covars,
    num_wts, 'PC1_HIST')

# Plot PC1 - simulation
xds_PCs_sim = xds_PCs.sel(
    time=slice(time_sim[0],time_sim[-1])
)
time_sim_covars = [x2d(t) for t in xds_PCs_sim.time]
Plot_Covariate(
    bmus_values_sim, xds_PCs_sim.PC1.values,
    time_sim, time_sim_covars,
    num_wts, 'PC1-SIM',
    num_sim,
)


# compare PCs
Plot_Compare_Covariate(
    num_wts,
    bmus_values_sim, time_sim,
    bmus_values_hist, time_hist,
    xds_PCs_sim.PC1.values, time_sim_covars,
    xds_PCs_hist.PC1.values, time_hist_covars,
    'PC1',
    n_sim = num_sim
)

Plot_Compare_Covariate(
    num_wts,
    bmus_values_sim, time_sim,
    bmus_values_hist, time_hist,
    xds_PCs_sim.PC2.values, time_sim_covars,
    xds_PCs_hist.PC2.values, time_hist_covars,
    'PC2',
    n_sim = num_sim
)

Plot_Compare_Covariate(
    num_wts,
    bmus_values_sim, time_sim,
    bmus_values_hist, time_hist,
    xds_PCs_sim.PC3.values, time_sim_covars,
    xds_PCs_hist.PC3.values, time_hist_covars,
    'PC3',
    n_sim = num_sim
)

# TODO: plot mjo comparison

