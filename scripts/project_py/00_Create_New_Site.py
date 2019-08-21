#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import ast
import pickle
import sys
sys.path.insert(0, op.dirname(os.getcwd()))

# pip
import numpy as np
import xarray as xr
from datetime import datetime

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../')

# tk 
from teslakit.project_site import Site


# --------------------------------------
# Create new teslakit project Site

data_folder = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
site = Site(data_folder, 'KWAJALEIN_TEST')

# Create empty folders
site.MakeDirs()


# --------------------------------------
# Required external input files:

# 01_SST_AnnualWeatherTypes
#    - DB.SST.hist_pacific                # SST Pacific area 'SST_1854_2017_Pacific.nc'

# 02_MJO_ALR_Simulation
#    - DB.MJO.hist                        # historical MJO (teslakit_downloader)

# 03_STORMS_Historical_ExtractRadius
#    - DB.TCs.noaa                        # NOAA WMO TCs (teslakit_downloader)

# 04_STORMS_Historical_CopulaSimulation_MDA

# 05_STORMS_Historical_RBFs_Interpolation

# 06_STORMS_CategoryChange_Probs
#    - DB.TCs.nakajo_mats                 # Nakajo synthetic TCs

# 07_WAVES_CalculateFamilies_RemoveTCs
#    - ST.WAVES.partitions_p1             # waves partitions data (GOW)

# 08_SLP_ESTELA_PCA_KMArg
#    - ST.WAVES.partitions_p1             # waves partitions data (GOW)
#    - ST.ESTELA.coastmat                 # estela coast (.mat)
#    - ST.ESTELA.estelamat                # estela data (.mat)
#    - ST.ESTELA.gowpoint                 # gow point (.mat)
#    - ST.ESTELA.slp                      # site slp data (.nc) extracted from CFS

# 09_SLP_ESTELA_ALR_Covariates
#    - DB.MJO.hist                        # historical MJO (teslakit_downloader)

# 10_Climate_Emulator


# --------------------------------------
# Files Summary

# each site folder contains two .ini files:
# - files.ini       a list of all the files that will be generated or read by teslakit project
# - parameters.ini  a list of all the parameters that will be used at teslakit calculations

# about required external files:
# many of the files in files.ini are generated while calculating a teslakit project, but waves partitions point and
# ESTELA are required as external output
#
# remaining external input are: Pacific SST historical, MJO historical, NOAA WMO TCs and Nakajo synthetic TCs


