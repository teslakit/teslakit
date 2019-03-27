#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import ast
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import numpy as np
import xarray as xr
from datetime import datetime

# tk libs
from lib.objs.tkpaths import Site
from lib.io.matlab import ReadGowMat
from lib.waves import GetDistribution


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')

DB = site.pc.DB                        # common database
ST = site.pc.site                      # site database
PR = site.params                       # site parameters

# input files
p_wvs_parts = ST.WAVES.partitions_p1
p_hist_r2_params = ST.TCs.hist_r2_params # hist storms parameters

# output files
p_wvs_parts_noTCs = ST.WAVES.partitions_notcs
p_wvs_fams_noTCs = ST.WAVES.families_notcs
p_wvs_fams_TCs_categ = ST.WAVES.families_tcs_categ

# wave families sectors
wvs_sectors = ast.literal_eval(PR.WAVES.sectors)

# date limits for TCs removal from waves data, and TC time window (hours)
tc_rm_date1 = PR.WAVES.tc_remov_date1
tc_rm_date2 = PR.WAVES.tc_remov_date2
tc_time_window = int(PR.WAVES.tc_remov_timew)


# --------------------------------------
# load wave point partitions data
xds_wvs_pts = ReadGowMat(p_wvs_parts)

# calculate wave families at sectors 
print('\nCalculating waves families in sectors: {0}'.format(wvs_sectors))
xds_wvs_fam = GetDistribution(xds_wvs_pts, wvs_sectors)


# --------------------------------------
# load historical storms-parameters inside r2 
xds_TCs_r2_params = xr.open_dataset(p_hist_r2_params)

# remove TCs before 1979 and after 2015
print(
'\nRemoving Historical TCs from waves partitions and families ...\n \
date_ini = {0}\n date_end = {1}\n time_window(h) = {2}'.format(
    tc_rm_date1, tc_rm_date2, tc_time_window)
)

d1 = np.datetime64(tc_rm_date1)
d2 = np.datetime64(tc_rm_date2)
dmin_dates = xds_TCs_r2_params.dmin_date.values

p1 = np.where(dmin_dates >= d1)[0][0]
p2 = np.where(dmin_dates <= d2)[0][-1]
xds_TCs_r2_params = xds_TCs_r2_params.sel(storm = slice(p1, p2))

# storms inside circle
storms = xds_TCs_r2_params.storm.values[:]


# for each storm: find hs_max instant and clean waves data inside "hs_max window"
xds_wvs_pts_noTCs = xds_wvs_pts.copy()
xds_wvs_fam_noTCs = xds_wvs_fam.copy()

# store removed waves (TCs window) families inside a storm category dictionary
d_wvs_fam_cats = {'{0}'.format(k):[] for k in range(6)}

for s in storms:

    # storm dates at dist_min and storm_end
    xds_s = xds_TCs_r2_params.sel(storm = slice(s, s))
    date_dmin = xds_s.dmin_date.values[0]
    date_last = xds_s.last_date.values[0]
    cat = xds_s.category.values[0]

    xds_wvs_s = xds_wvs_pts.sel(
        time = slice(str(date_dmin), str(date_last))
    )

    # get hs_max date 
    t_hs_max = xds_wvs_s.where(
        xds_wvs_s.hs ==  xds_wvs_s.hs.max(), drop=True
    ).time.values[:][0]

    # hs_max time window 
    w1 = t_hs_max - np.timedelta64(tc_time_window,'h')
    w2 = t_hs_max + np.timedelta64(tc_time_window,'h')

    # clean waves partitions 
    xds_wvs_pts_noTCs = xds_wvs_pts_noTCs.where(
        (xds_wvs_pts_noTCs.time < w1) |
        (xds_wvs_pts_noTCs.time > w2)
    )
    # clean waves families
    xds_wvs_fam_noTCs = xds_wvs_fam_noTCs.where(
        (xds_wvs_fam_noTCs.time < w1) |
        (xds_wvs_fam_noTCs.time > w2)
    )

    # store waves families inside storm-removal-window
    xds_fams_s = xds_wvs_fam.sel(time = slice(w1, w2))
    d_wvs_fam_cats['{0}'.format(cat)].append(xds_fams_s)

# merge removal-window waves families data
for k in d_wvs_fam_cats.keys():
    d_wvs_fam_cats[k] = xr.merge(d_wvs_fam_cats[k])


# store results
xds_wvs_pts_noTCs.to_netcdf(p_wvs_parts_noTCs, 'w')
xds_wvs_fam_noTCs.to_netcdf(p_wvs_fams_noTCs, 'w')
print('\nWaves Partitions (TCs removed) stored at:\n{0}'.format(p_wvs_parts_noTCs))
print(xds_wvs_pts_noTCs)
print('\nWaves Families   (TCs removed) stored at:\n{0}'.format(p_wvs_fams_noTCs))
print(xds_wvs_fam_noTCs)

# one file for each waves_families - category
for k in d_wvs_fam_cats.keys():
    p_s = op.join(p_wvs_fams_TCs_categ, 'waves_fams_cat{0}.nc'.format(k))
    d_wvs_fam_cats[k].to_netcdf(p_s, 'w')
print('\nWaves Families   (TCs windows) stored at:\n{0}'.format(p_wvs_fams_TCs_categ))

