#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress
from datetime import datetime

# tk libs
from lib.objs.tkpaths import PathControl
from lib.io.matlab import ReadMareografoMat
from lib.tides import Calculate_MMSL
from lib.statistical import runmean


# --------------------------------------
# data storage and path control
pc = PathControl()
pc.SetSite('test_site')

# Load data from tide gauge
xds_tide = ReadMareografoMat(pc.site.tds.mareografo)

# fix data 
xds_tide.rename(
    {'WaterLevel':'tide'},
    inplace=True)
xds_tide['tide'] = xds_tide['tide']*1000

# calculate SLR using linear regression
time = np.array(range(len(xds_tide.time.values[:])))
tide = xds_tide.tide.values[:]

slope, intercept, r_value, p_value, std_err = linregress(time,tide)
slr = intercept + slope * time

# remove slr from tide 
tide = tide - slr

# remove tide running mean
time_window = 365*24*3
tide = tide - runmean(tide, time_window, 'mean')


# calculate mmsl
xds_tide['tide'].values = tide
xds_MMSL = Calculate_MMSL(xds_tide, 1996, 2017)


# Load SST Anual Weather Types PCs
xds_KMA = xr.open_dataset(pc.site.sst_awt.xds_KMA)
PCs = np.array(xds_KMA.PCs.values)
PC1 = PCs[:,0]
PC2 = PCs[:,1]
PC3 = PCs[:,2]
PCs_years = [pd.to_datetime(dt).year for dt in xds_KMA.time.values]
MMSL_time = xds_MMSL.time.values
MMSL = xds_MMSL.mmsl.values

# PCs calculations
ntrs_m_mean = np.array([])
ntrs_time = []

for y in PCs_years:
    pos = np.where(
        (MMSL_time >= np.datetime64('{0}-06-01'.format(y))) &
        (MMSL_time <= np.datetime64('{0}-05-29'.format(y+1)))
    )

    if pos[0].size:
        ntrs_m_mean = np.concatenate((ntrs_m_mean, MMSL[pos]),axis=0)
        ntrs_time.append(MMSL_time[pos])

    # TODO: PC PADDING
    # SACAR PC1,PC2,PC3 DE MMSL

ntrs_time = np.concatenate(ntrs_time)



# TODO: CREAR MODELO REGRESION LINEAL B1+B2X2+B3X3...
# ALIMENTARLO CON PC1,PC2,PC3


# TODO: PREDECIR 1000 YEARS CON MODELO REGRESION LINEAL

