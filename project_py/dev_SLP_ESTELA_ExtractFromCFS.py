#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# tk libs
from lib.objs.tkpaths import Site
from lib.io.cfs import ReadSLP


# --------------------------------------
# Site paths and parameters
site = Site('KWAJALEIN')

DB = site.pc.DB                            # common database
ST = site.pc.site                          # site database
PR = site.params                           # site parameters

# input files
p_DB_cfs_prmsl = DB.CFS.cfs_prmsl

# output files
p_site_SLP =  ST.ESTELA.slp

# SLP extraction coordinates 
lat1 = float(PR.SLP.lat1)
lat2 = float(PR.SLP.lat2)
lon1 = float(PR.SLP.lon1)
lon2 = float(PR.SLP.lon2)
resample = int(PR.SLP.resample)  # 2º resolution


# --------------------------------------
# load predictor data (SLP) from CFSR and save to .nc 

print('\nReading SLP data from CFS_prmsl database...')
xds_SLP_site = ReadSLP(
    p_DB_cfs_prmsl,
    lat1, lat2, lon1, lon2, resample,
    p_save=p_site_SLP)

