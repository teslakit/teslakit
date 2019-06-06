#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..','..'))

# python libs
import matplotlib.pyplot as plt

# custom libs
from teslakit.project_site import Site
from teslakit.climate_emulator import Climate_Emulator


# --------------------------------------
# Test data storage
site = Site('KWAJALEIN')
ST = site.pc.site                      # site database

# input files
p_ce = ST.EXTREMES.climate_emulator    # climate emulator folder


# --------------------------------------
# Load climate emulator
CE = Climate_Emulator(p_ce)
CE.Load()

# test Report_Fit
CE.Report_Fit()

plt.show()

