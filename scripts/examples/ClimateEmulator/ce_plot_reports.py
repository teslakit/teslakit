#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op

# python libs
import matplotlib.pyplot as plt

# DEV: override installed teslakit
import sys
sys.path.insert(0,'../../../')

# teslakit
from teslakit.project_site import PathControl
from teslakit.climate_emulator import Climate_Emulator


# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ClimateEmulator', 'CE_FitExtremes')

# input files
p_ce = op.join(p_test, 'ce')  # climate emulator


# --------------------------------------
# Load climate emulator
CE = Climate_Emulator(p_ce)
CE.Load()

# test Report_Fit
CE.Report_Fit()

plt.show()

