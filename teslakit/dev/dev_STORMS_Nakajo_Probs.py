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

# tk libs
from lib.objs.tkpaths import Site
from lib.io.matlab import ReadNakajoMats
from lib.util.operations import GetUniqueRows
from lib.tcyclone import Extract_Circle, GetStormCategory, \
SortCategoryCount


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')

DB = site.pc.DB                        # common database
ST = site.pc.site                      # site database
PR = site.params                       # site parameters

# input files
p_nakajo_mats = DB.TCs.nakajo_mats

# output files
p_probs_synth = ST.TCs.probs_synth

# wave point lon, lat, and radius for TC selection
pnt_lon = float(PR.WAVES.point_longitude)
pnt_lat = float(PR.WAVES.point_latitude)
r1 = float(PR.TCS.r1)                   # bigger one
r2 = float(PR.TCS.r2)                   # smaller one


# --------------------------------------
# read each nakajo simulation pack from .mat custom files 
xds_Nakajo = ReadNakajoMats(p_nakajo_mats)

# rename lon,lat variables 
xds_Nakajo.rename(
    {
        'ylon_TC':'lon',
        'ylat_TC':'lat',
        'yCPRES':'pressure',
    }, inplace=True)


# Extract synthetic TCs at 2 radius to get category change 
print(
'\nExtracting Synthetic TCs from Nakajo database...\n \
Lon = {0:.2f}º , Lat = {1:.2f}º\n \
R1  = {2:6.2f}º , R2  = {3:6.2f}º'.format(
    pnt_lon, pnt_lat, r1, r2)
)

# Extract TCs inside R=14 and positions
_, xds_in_r1 = Extract_Circle(
    xds_Nakajo, pnt_lon, pnt_lat, r1)

# Extract TCs inside R=4 and positions
_, xds_in_r2 = Extract_Circle(
    xds_Nakajo, pnt_lon, pnt_lat, r2)


print('\nCalculating Syntethic TCs category-change probabilities...')

# Get min pressure and storm category inside both circles
n_storms = len(xds_in_r1.storm)
categ_r1r2 = np.empty((n_storms, 2))
for i in range(len(xds_in_r1.storm)):

    # min pressure inside R1
    storm_in_r1 = xds_in_r1.isel(storm=[i])
    storm_id = storm_in_r1.storm.values[0]
    storm_cat_r1 = storm_in_r1.category

    # min pressure inside R2
    if storm_id in xds_in_r2.storm.values[:]:
        storm_in_r2 = xds_in_r2.sel(storm=[storm_id])
        storm_cat_r2 = storm_in_r2.category
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

# store output using xarray
xds_categ_cp = xr.Dataset(
    {
        'category_change_count': (('category','category'), m_count[:-1,:]),
        'category_change_sum': (('category'), m_count[-1,:]),
        'category_change_probs': (('category','category'), probs[:-1,:]),
        'category_nochange_probs': (('category'), probs[-1,:]),
        'category_change_cumsum': (('category','category'), probs_cs[:-1,:]),
    },
    coords = {
        'category': [0,1,2,3,4,5]
    }
)
xds_categ_cp.to_netcdf(p_probs_synth)
print xds_categ_cp
print('\nSyntethic TCs category-change stored at:\n{0}'.format(p_probs_synth))

