#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

# custom libs
from ttide.t_tide import t_tide
from ttide.t_predic import t_predic

# tk libs
from lib.objs.tkpaths import Site
from lib.io.aux_nc import StoreBugXdset as sbxds
from lib.plotting.tides import Plot_AstronomicalTide, Plot_ValidateTTIDE

# TODO: ACTUALIZAR A LIBRERIA UTIDE
# https://www.eoas.ubc.ca/~rich/#T_Tide


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')

DB = site.pc.DB                        # common database
ST = site.pc.site                      # site database
PR = site.params                       # site parameters

# input files
p_astro_fit = ST.TIDE.hist_astro

# output files
p_astro_sim = ST.TIDE.sim_astro

# export figs
p_export_tds = ST.export_figs.tds

# Simulation dates
d1_sim = np.datetime64(PR.SIMULATION.date_ini)
d2_sim = np.datetime64(PR.SIMULATION.date_end)


# --------------------------------------
# load astronomical tide data
xds_atide = xr.open_dataset(p_astro_fit)
xds_atide.rename(
    {'observed':'level',
     'predicted':'tide',
    }, inplace=True)

# remove tide nanmin
xds_atide['tide'] = xds_atide.tide - np.nanmin(xds_atide.tide)

# data length has to be lesser than 18.6 years
dt_cut = np.datetime64('1998-06-01')
xds_atide = xds_atide.where(xds_atide.time >= dt_cut, drop=True)

# Plot astronomical tide
time = xds_atide.time.values[:]
tide = xds_atide.tide.values[:]
p_export = op.join(p_export_tds, 'astronomical_tide.png')
Plot_AstronomicalTide(time, tide, p_export)


# --------------------------------------
# t_tide library - Fit

# TODO: ttide ha de se calibrada con un year de datos (no 18.6)
lat0 = 9.75
d_out = t_tide(xds_atide.tide.values, dt=1, lat=np.array(lat0))

# variables used for prediction
names = d_out['nameu']
freq = d_out['fu']
tidecon = d_out['tidecon']


# --------------------------------------
# t_tide library - Validation

d1_val = xds_atide.time.values[0]
d2_val = xds_atide.time.values[-1]
date_val = np.arange(d1_val, d2_val, dtype='datetime64[h]')
tide_tt = t_predic(
    date_val, names, freq, tidecon,
    lat=lat0, ltype='nodal')

# Plot ttide validation 
time = xds_atide.time.values[:-1]
tide = xds_atide.tide.values[:-1]
tide = tide - np.nanmean(tide)
p_export = op.join(p_export_tds, 'ttide_validation.png')
Plot_ValidateTTIDE(time, tide, tide_tt, p_export)


# --------------------------------------
# t_tide library - Prediction
date_pred = np.arange(d1_sim, d2_sim, dtype='datetime64[h]')
atide_pred = t_predic(
    date_pred, names, freq, tidecon,
    lat=lat0, ltype='nodal')


# Store data
dt_pred = [d.astype(datetime) for d in date_pred]
xds_atide_sim = xr.Dataset(
    {
        'tide'   :(('time',), atide_pred),
    },
    {'time' : dt_pred}
)

# xarray.Dataset.to_netcdf() wont work with this time array and time dtype
sbxds(xds_atide_sim, p_astro_sim)
print('\nAstronomical Tide Simulation stored at:\n{0}\n'.format(p_astro_sim))

# Plot astronomical tide prediction
time = xds_atide_sim.time.values[:]
tide = xds_atide_sim.tide.values[:]
p_export = op.join(p_export_tds, 'astronomical_tide_pred.png')
Plot_AstronomicalTide(time, tide, p_export)

