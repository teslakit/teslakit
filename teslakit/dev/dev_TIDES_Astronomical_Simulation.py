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
import netCDF4
from datetime import datetime

# custom libs
from ttide.t_tide import t_tide
from ttide.t_predic import t_predic

# tk libs
from lib.objs.tkpaths import Site
from lib.io.matlab import ReadAstroTideMat

# TODO: ACTUALIZAR A LIBRERIA UTIDE
# https://www.eoas.ubc.ca/~rich/#T_Tide


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')
site.Summary()

# input files
# TODO: REVISAR/MODIFICAR DATOS TIDE Y USAR .NC
p_tide_astro = site.pc.site.tds.MAR_1820000

# output files
p_astro_sim = site.pc.site.tds.sim_astro

# Simulation dates
d1_sim = np.datetime64(site.params.SIMULATION.date_ini)
d2_sim = np.datetime64(site.params.SIMULATION.date_end)


# --------------------------------------
# load astronomical tide data
xds_atide = ReadAstroTideMat(p_tide_astro)
xds_atide.rename(
    {'observed':'level',
     'predicted':'tide',
    }, inplace=True)

# remove tide nanmin
xds_atide['tide'] = xds_atide.tide - np.nanmin(xds_atide.tide)

# TODO: TTIDE LIBRARY NO PREPARADA PARA MANEJAR MAS DE 18.6 YEARS
dt_cut = np.datetime64('1998-06-01')
xds_atide = xds_atide.where(xds_atide.time>=dt_cut,drop=True)


# --------------------------------------
# t_tide library - Fit

lat0 = 9.75
d_out = t_tide(xds_atide.tide.values, dt=1, lat=np.array(lat0))
# TODO: ttide ha de se calibrada con un year de datos (no 18.6)

# variables used for prediction
names = d_out['nameu']
freq = d_out['fu']
tidecon = d_out['tidecon']


# --------------------------------------
# t_tide library - Prediction
date_pred = np.arange(d1_sim, d2_sim, dtype='datetime64[h]')
atide_pred = t_predic(
    date_pred, names, freq, tidecon,
    lat=lat0, ltype='nodal')


# --------------------------------------
# use netCDF4 library to store this data
calendar = 'standard'
units = 'days since 1970-01-01 00:00:00'
times = [z.astype(datetime) for z in date_pred]

# open file
root = netCDF4.Dataset(p_astro_sim, 'w', format='NETCDF4')
root.createDimension('time', len(date_pred))

# time variable
timevar = root.createVariable(
    varname='time',dimensions=('time',), datatype='float32')
timevar[:] = netCDF4.date2num(times, units=units, calendar=calendar)
timevar.units = units

# astronomical tide variable
atvar = root.createVariable(
    varname='astronomical_tide', dimensions=('time',), datatype='float32')
atvar[:] = atide_pred

# close file
root.close()

print('\nAstronomical Tide Simulation stored at:\n{0}\n'.format(p_astro_sim))


# --------------------------------------
# TODO: t_tide library validation
validate_ttide = False
if validate_ttide:

    # astronomical tide prediction
    # TODO: COGER ESTAS FECHAS AUTO, ES PARA COMPARAR
    dp1 = np.datetime64('1998-06-01')
    dp2 = np.datetime64('2016-12-31')
    date_pred = np.arange(dp1, dp2, dtype='datetime64[h]')

    atide_pred = t_predic(
        date_pred, names, freq, tidecon,
        lat=lat0,ltype='nodal')


    # compare astronomical tide data and prediction
    atide_nanmean = xds_atide.tide - np.nanmean(xds_atide.tide)

    # TODO: ADD PLOTS
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(xds_atide.time.values, atide_nanmean, 'k-', linewidth=0.5, label='data')
    ax.plot(date_pred, atide_pred, 'r--', linewidth=0.5, label='model')
    ax.set_xlim([np.datetime64('1998-06-01'), np.datetime64('2000-01-01')])
    ax.legend()
    plt.show()



