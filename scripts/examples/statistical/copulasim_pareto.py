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
from teslakit.statistical import CopulaSimulation


# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')


# --------------------------------------
# load data and set parameters

_, xds_TCs_r2_params = db.Load_TCs_r2()  # TCs parameters inside radius 2

# TCs random generation and MDA parameters
num_sim_rnd = 100000
num_sel_mda = 1000


# --------------------------------------
# Probabilistic simulation Historical TCs

def FixPareto(var):
    'Fix data. It needs to start at 0 for Pareto adjustment '
    var = var.astype(float)
    mx = np.amax(var)
    aux = mx + np.absolute(var - mx)
    var_pareto = aux - np.amin(aux) + 0.00001

    return var_pareto, np.amin(aux)

# aux function
def adjust_to_pareto(var):
    'Fix data. It needs to start at 0 for Pareto adjustment '
    var = var.astype(float)
    var_pareto =  np.amax(var) - var + 0.00001

    return var_pareto

def adjust_from_pareto(var_base, var_pareto):
    'Returns data from pareto adjustment'

    var = np.amax(var_base) - var_pareto + 0.00001

    return var

# use small radius parameters (4º)
pmean = xds_TCs_r2_params.pressure_mean.values[:]

print('pmean')
print(pmean)
print()

# correct ouput
pmean_pc, _ = FixPareto(pmean)

print('pmean pareto good')
print(pmean_pc)
print()


# fix pressure for p
pmean_p = adjust_to_pareto(pmean)

print('pmean_pareto')
print(pmean_p)
print()


# BACK from pareto fix
pmean_bk = adjust_from_pareto(pmean, pmean_p)

print('pmean_back')
print(pmean_bk)
print()




