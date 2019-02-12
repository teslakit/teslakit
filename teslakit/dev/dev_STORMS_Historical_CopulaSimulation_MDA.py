#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [18, 8]

# tk libs
from lib.objs.tkpaths import Site
from lib.data_fetcher.STORMS import Download_NOAA_WMO
from lib.tcyclone import Extract_Circle
from lib.plotting.storms import WorldMap_Storms
from lib.statistical import CopulaSimulation
from lib.MDA import MaxDiss_Simplified_NoThreshold


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')

DB = site.pc.DB                        # common database
ST = site.pc.site                      # site database
PR = site.params                       # site parameters

# input files
p_hist_r1_params = ST.TCs.hist_r1_params
p_hist_r2_params = ST.TCs.hist_r2_params

# output files
p_sim_r2_params = ST.TCs.sim_r2_params
p_mda_r2_params = ST.TCs.mda_r2_params

# parameters
num_sim_rnd = int(PR.TCS.num_simulate_rnd)
num_sel_mda = int(PR.TCS.num_select_mda)


# --------------------------------------
# Probabilistic simulation Historical TCs

# aux function
def FixPareto(var):
    'Fix data. It needs to start at 0 for Pareto adjustment '
    var = var.astype(float)
    mx = np.amax(var)
    aux = mx + np.absolute(var - mx)
    var_pareto = aux - np.amin(aux) + 0.00001

    return var_pareto, np.amin(aux)


# use small radius (4º)
xds_TCs_r2_params = xr.open_dataset(p_hist_r2_params)

pmean = xds_TCs_r2_params.pressure_mean.values[:]
pmin = xds_TCs_r2_params.pressure_min.values[:]
gamma = xds_TCs_r2_params.gamma.values[:]
delta = xds_TCs_r2_params.delta.values[:]
vmean = xds_TCs_r2_params.velocity_mean.values[:]


# fix pressure for pareto adjustment (start at 0)
pmean_p, auxmin_pmean = FixPareto(pmean)
pmin_p, auxmin_pmin = FixPareto(pmin)
# join storm parameters for copula simulation
storm_params = np.column_stack(
    (pmean_p, pmin_p, gamma, delta, vmean)
)

# statistical simulate PCs using copulas
kernels = ['GPareto', 'GPareto', 'ECDF', 'ECDF', 'ECDF']
storm_params_sim = CopulaSimulation(storm_params, kernels, num_sim_rnd)

# store simulated storms - parameters
xds_TCs_r2_sim_params = xr.Dataset(
    {
        'pressure_mean':(('storm'), storm_params_sim[:,0] + auxmin_pmean),
        'pressure_min':(('storm'), storm_params_sim[:,1] + auxmin_pmin),
        'gamma':(('storm'), storm_params_sim[:,2]),
        'delta':(('storm'), storm_params_sim[:,3]),
        'velocity_mean':(('storm'), storm_params_sim[:,4]),
    },
    coords = {
        'storm':(('storm'), np.arange(num_sim_rnd))
    },
)
xds_TCs_r2_sim_params.to_netcdf(p_sim_r2_params)

print(xds_TCs_r2_sim_params)
print('\n\nSimulated TCs parameters stored at:\n{0}'.format(
    p_sim_r2_params))


# --------------------------------------
# MaxDiss classification

# get simulated parameters  
pmean_s = xds_TCs_r2_sim_params.pressure_mean.values[:]
pmin_s = xds_TCs_r2_sim_params.pressure_min.values[:]
gamma_s = xds_TCs_r2_sim_params.gamma.values[:]
delta_s = xds_TCs_r2_sim_params.delta.values[:]
vmean_s = xds_TCs_r2_sim_params.velocity_mean.values[:]

# subset, scalar and directional indexes
data_mda = np.column_stack((pmean_s, pmin_s, vmean_s, delta_s, gamma_s))
ix_scalar = [0,1,2]
ix_directional = [3,4]

centroids = MaxDiss_Simplified_NoThreshold(
    data_mda, num_sel_mda, ix_scalar, ix_directional
)

# store MDA storms - parameters 
xds_TCs_r2_MDA_params = xr.Dataset(
    {
        'pressure_mean':(('storm'), centroids[:,0]),
        'pressure_min':(('storm'), centroids[:,1]),
        'gamma':(('storm'), centroids[:,2]),
        'delta':(('storm'), centroids[:,3]),
        'velocity_mean':(('storm'), centroids[:,4]),
    },
    coords = {
        'storm':(('storm'), np.arange(num_sel_mda))
    },
)
xds_TCs_r2_MDA_params.to_netcdf(p_mda_r2_params)

print(xds_TCs_r2_MDA_params)
print('\n\nMaxDiss TCs parameters stored at:\n{0}'.format(
    p_mda_r2_params))

