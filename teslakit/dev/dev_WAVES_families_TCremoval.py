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
from datetime import datetime

# tk libs
from lib.objs.tkpaths import PathControl
from lib.io.matlab import ReadGowMat
from lib.waves import GetDistribution
from lib.tcyclone import Extract_Circle


# --------------------------------------
# data storage and path control
pc = PathControl()
pc.SetSite('test_site')


# --------------------------------------
# load wave point partitions data
xds_wvs_pts = ReadGowMat(pc.site.wvs.partitions_p1)

# calculate wave families at sectors 
sectors = [(210, 22.5), (22.5, 135)]
xds_wvs_fam = GetDistribution(xds_wvs_pts, sectors)


# --------------------------------------
# load historical TCs
xds_wmo_fix = xr.open_dataset(pc.DB.tcs.noaa_fix)

# extract TCs inside circle using GOW point as center 
p_lon = xds_wvs_pts.lon
p_lat = xds_wvs_pts.lat
r = 4

_, xds_in = Extract_Circle(
    xds_wmo_fix, p_lon, p_lat, r)

# remove TCs before 1979 and after 2015
d1 = np.datetime64('1979-01-01')
d2 = np.datetime64('2015-12-31')
dmin_dates = xds_in.dmin_date.values

p1 = np.where(dmin_dates >= d1)[0][0]
p2 = np.where(dmin_dates <= d2)[0][-1]
xds_in = xds_in.sel(storm = slice(p1, p2))

# storms inside circle
storms = xds_in.storm.values[:]


# for each storm: find hs_max instant and clean waves data inside "hs_max windows"
window = 12  # hours
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
    w1 = t_hs_max - np.timedelta64(window,'h')
    w2 = t_hs_max + np.timedelta64(window,'h')

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
xds_wvs_pts_noTCs.to_netcdf(pc.site.wvs.partitions_noTCs,'w')
xds_wvs_fam_noTCs.to_netcdf(pc.site.wvs.families_noTCs,'w')

print xds_wvs_fam_noTCs
# TODO: los datos de oleaje (particiones y familias) estan limpios
# hay que hacer algo mas con estos intervalos hs_peak?


