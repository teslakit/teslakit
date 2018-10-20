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
from lib.tcyclone import Extract_Circle


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')
site.Summary()

# input files
p_wvs_parts = site.pc.site.wvs.partitions_p1
p_hist_tcs = site.pc.DB.tcs.noaa_fix

# output files
p_wvs_parts_noTCs = site.pc.site.wvs.partitions_noTCs
p_wvs_fams_noTCs = site.pc.site.wvs.families_noTCs

# wave point lon, lat, families sectors, and radius for TCs selection
wvs_sectors = ast.literal_eval(site.params.WAVES.sectors)
pnt_lon = float(site.params.WAVES.point_longitude)
pnt_lat = float(site.params.WAVES.point_latitude)
r2 = float(site.params.TCS.r2)   # smaller one

# also date limits for TCs removal from waves data, and TC time window (hours)
tc_rm_date1 = site.params.WAVES.tc_remov_date1
tc_rm_date2 = site.params.WAVES.tc_remov_date2
tc_time_window = int(site.params.WAVES.tc_remov_timew)


# --------------------------------------
# load wave point partitions data
xds_wvs_pts = ReadGowMat(p_wvs_parts)

# calculate wave families at sectors 
print('\nCalculating waves families in sectors: {0}'.format(wvs_sectors))
xds_wvs_fam = GetDistribution(xds_wvs_pts, wvs_sectors)


# --------------------------------------
# load historical TCs
xds_hist_tcs = xr.open_dataset(p_hist_tcs)

# extract TCs inside circle using GOW point as center 
print(
'\nExtracting Historical TCs from WMO database...\n \
Lon = {0:.2f}º , Lat = {1:.2f}º, R2  = {2:6.2f}º'.format(
    pnt_lon, pnt_lat, r2)
)

_, xds_in = Extract_Circle(
    xds_hist_tcs, pnt_lon, pnt_lat, r2)


# remove TCs before 1979 and after 2015
print(
'\nRemoving Historical TCs from waves partitions and families ...\n \
date_ini = {0}\n date_end = {1}\n time_window(h) = {2}'.format(
    tc_rm_date1, tc_rm_date2, tc_time_window)
)

d1 = np.datetime64(tc_rm_date1)
d2 = np.datetime64(tc_rm_date2)
dmin_dates = xds_in.dmin_date.values

p1 = np.where(dmin_dates >= d1)[0][0]
p2 = np.where(dmin_dates <= d2)[0][-1]
xds_in = xds_in.sel(storm = slice(p1, p2))

# storms inside circle
storms = xds_in.storm.values[:]


# for each storm: find hs_max instant and clean waves data inside "hs_max window"
xds_wvs_pts_noTCs = xds_wvs_pts.copy()
xds_wvs_fam_noTCs = xds_wvs_fam.copy()

for s in storms:

    # storm dates at dist_min and storm_end
    xds_s = xds_in.sel(storm = slice(s, s))
    date_dmin = xds_s.dmin_date.values[0]
    date_last = xds_s.last_date.values[0]

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

# store results
xds_wvs_pts_noTCs.to_netcdf(p_wvs_parts_noTCs, 'w')
xds_wvs_fam_noTCs.to_netcdf(p_wvs_fams_noTCs, 'w')
print('\nWaves Partitions (TCs removed) stored at:\n{0}'.format(p_wvs_parts_noTCs))
print('\nWaves Families   (TCs removed) stored at:\n{0}'.format(p_wvs_fams_noTCs))

# TODO hay que hacer algo mas con estos intervalos hs_peak?


