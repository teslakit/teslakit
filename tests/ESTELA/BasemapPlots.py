#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

#pip
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import xarray as xr


# --------------------------------------
# Test data storage

pc = PathControl()
p_tests = pc.p_test_data
p_test = op.join(p_tests, 'ESTELA', 'Roi_Kwajalein')
p_SLP_save = op.join(p_test, 'SLP.nc')


# --------------------------------------
# load and use xarray saved predictor data (faster)
xds_SLP_site = xr.open_dataset(p_SLP_save)

# test plots

# basemap: ortho
m = Basemap(
    projection='ortho',
    lon_0=125,lat_0=0,
    resolution='l',
)

# simple map: coastlines
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary(fill_color='aqua')

# draw data
XX,YY = np.meshgrid(
    xds_SLP_site.longitude.values,
    xds_SLP_site.latitude.values,
)
DD = xds_SLP_site.isel(time=5).SLP.values
m.contourf(XX,YY,DD)

# draw parallels.
parallels = np.arange(-90.,90.,10.)
m.drawparallels(parallels, labels=[1,0,0,0],fontsize=6)
# draw meridians
meridians = np.arange(0.,360.,10.)
m.drawmeridians(meridians, labels=[0,0,0,1],fontsize=6)

plt.show()

