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
import datetime

# custom libs
from ttide.t_tide import t_tide
from ttide.t_predic import t_predic

# tk libs
from lib.objs.tkpaths import PathControl
from lib.io.matlab import ReadAstroTideMat
from lib.tides import fun  # TODO


# --------------------------------------
# data storage and path control
pc = PathControl()
pc.SetSite('test_site')


# load astronomical tide data
xds_atide = ReadAstroTideMat(pc.site.tds.MAR_1820000)
xds_atide.rename(
    {'observed':'level',
     'predicted':'tide',
    }, inplace=True)

# remove tide nanmin
xds_atide['tide'] = xds_atide.tide - np.nanmin(xds_atide.tide)

# TODO: TTIDE LIBRARY NO PREPARADA PARA MANEJAR MAS DE 18.6 YEARS
dt_cut = np.datetime64('1998-06-01')
xds_atide = xds_atide.where(xds_atide.time>=dt_cut,drop=True)

# use ttide function
lat0 = 9.75
d_out = t_tide(xds_atide.tide.values, dt=1, lat=np.array(lat0))

# variables used for prediction
names = d_out['nameu']
freq = d_out['fu']
tidecon = d_out['tidecon']


# astronomical tide prediction
dp1 = np.datetime64('1998-06-01')
dp2 = np.datetime64('2016-12-31')
date_pred = np.arange(dp1, dp2, dtype='datetime64[h]')

atide_pred = t_predic(
    date_pred, names, freq, tidecon,
    lat=lat0,ltype='nodal')


# compare astronomical tide data and prediction
atide_nanmean = xds_atide.tide - np.nanmean(xds_atide.tide)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(xds_atide.time.values, atide_nanmean, 'k-', linewidth=0.5, label='data')
ax.plot(date_pred, atide_pred, 'r--', linewidth=0.5, label='model')
ax.set_xlim([np.datetime64('1998-06-01'), np.datetime64('2000-01-01')])
ax.legend()
plt.show()


# TODO: predict 1000 years.
# necesitamos un objeto que controle los tiempos (y parametros) de todas las simulaciones para
#los distintos tramos de teslakit

# TODO: ADD PLOTS

