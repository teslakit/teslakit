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
from teslakit.plotting.awt import Plot_AWTs_DWTs_Probs



# --------------------------------------
# Teslakit database

p_data = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
db = Database(p_data)

# set site
db.SetSite('KWAJALEIN')


# Load AWTs (SST) and DWTs (ESTELA SLP)
AWT_hist, DWT_hist, AWT_sim, DWT_sim = db.Load_AWTs_DWTs_Plots(n_sim=0)

n_clusters_AWT = 6
n_clusters_DWT = 36


# Plot AWTs/DWTs Probs
Plot_AWTs_DWTs_Probs(
    AWT_hist, n_clusters_AWT,
    DWT_hist, n_clusters_DWT,
    ttl = 'DWTs Probabilities by AWTs - Historical'
)

Plot_AWTs_DWTs_Probs(
    AWT_sim, n_clusters_AWT,
    DWT_sim, n_clusters_DWT,
    ttl = 'DWTs Probabilities by AWTs - Simulation'
)

