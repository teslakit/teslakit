#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

import xarray as xr
import numpy as np

# tk libs
from lib.io.matlab import ReadNakajoMats
from lib.util.operations import GetUniqueRows
from lib.hurricanes import Extract_Circle, GetStormCategory, \
SortCategoryCount

# data storage
p_data = op.join(op.dirname(__file__), '..', 'data')
p_data_hurr = op.join(p_data, 'HURR')

# histoirical and synthetic hurricanes databases (input)
p_hurr_nakajo_mats = op.join(p_data_hurr, 'Nakajo_tracks')
p_hurr_nakajo_pk = op.join(p_data_hurr, 'Nakajo_tracks','Nakajo_tracks.nc')


# read each nakajo simulation pack from .mat custom files 
xds_Nakajo = ReadNakajoMats(p_hurr_nakajo_mats)

# rename lon,lat variables 
xds_Nakajo.rename(
    {
        'ylon_TC':'lon',
        'ylat_TC':'lat',
        'yCPRES':'pressure',
        'ySPEED':'wind_speed',  # TODO es wind speed?
    }, inplace=True)


# Select hurricanes that crosses a circular area 
p_lon = 167.5
p_lat = 9.5
r1 = 14  # degree
r2 = 4  # degree

# Extract storms inside R=14 and positions
xds_storms_r1, xds_inside_r1 = Extract_Circle(
    xds_Nakajo, p_lon, p_lat, r1)

# Extract storms inside R=4 and positions
xds_storms_r2, xds_inside_r2 = Extract_Circle(
    xds_Nakajo, p_lon, p_lat, r2)


# Get min pressure and storm category inside both circles
n_storms = len(xds_inside_r1.storm)
categ_r1r2 = np.empty((n_storms, 2))
for i in range(len(xds_inside_r1.storm)):

    # min pressure inside R1
    storm_in_r1 = xds_inside_r1.isel(storm=[i])
    storm_id = storm_in_r1.storm.values[0]
    storm_cat_r1 = storm_in_r1.inside_category

    # min pressure inside R2
    if storm_id in xds_inside_r2.storm.values[:]:
        storm_in_r2 = xds_inside_r2.sel(storm=[storm_id])
        storm_cat_r2 = storm_in_r2.inside_category
    else:
        storm_cat_r2 = 9  # no category 

    # store categories
    categ_r1r2[i,:] = [storm_cat_r1, storm_cat_r2]


# count category changes and sort it
categ_count = GetUniqueRows(categ_r1r2)
categ_count = SortCategoryCount(categ_count)

# calculate probability
m_count = np.reshape(categ_count[:,2], (6,-1)).T
m_sum = np.sum(m_count,axis=0)

probs = m_count.astype(float)/m_sum.astype(float)
probs_cs = np.cumsum(probs, axis=0)


print m_count
print ''
print probs
print ''
print probs_cs

