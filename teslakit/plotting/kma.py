#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
from math import sqrt
import itertools
import calendar
from datetime import datetime, timedelta

# pip
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy.interpolate import interp1d

# teslakit
from .custom_colors import colors_awt
from ..custom_dateutils import xds2datetime
from ..util.operations import GetDivisors

# import constants
from .config import _faspect, _fsize, _fdpi

# Weather Type colors
wt_colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']

def Plot_KMArg_clusters_datamean(xds_datavar, bmus, p_export=None):
    '''
    TODO: documentar correctamente
    '''

    # TODO: use better pcolor (like at pptx file)
    # TODO: draw land with fixed KMA cluster colors

    # get some data
    clusters = sorted(set(bmus))
    n_clusters = len(clusters)

    # TODO: AUTOMATIZAR VMIN VMAX
    #var_max = np.max(xds_datavar.values)
    #var_min = np.min(xds_datavar.values)
    var_min = 990
    var_max = 1030
    scale = 1/100.0  # scale from Pa to mbar

    # prepare figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    # Get number of rows and cols for gridplot 
    sqrt_clusters = sqrt(n_clusters)
    if sqrt_clusters.is_integer():
        n_rows = int(sqrt_clusters)
        n_cols = int(sqrt_clusters)
    else:
        l_div = GetDivisors(n_clusters)
        n_rows = l_div[len(l_div)/2]
        n_cols = n_clusters/n_rows

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.0)

    grid_row = 0
    grid_col = 0
    for ic in clusters:
        # data mean
        pos_cluster = np.where(bmus==ic)[0][:]
        mean_cluster = xds_datavar.isel(time=pos_cluster).mean(dim='time')

        # convert input units
        mean_cluster_scaled = np.multiply(mean_cluster, scale)

        # grid plot
        ax = plt.subplot(gs[grid_row, grid_col])
        pcm = ax.pcolormesh(
            np.flipud(mean_cluster_scaled.values),
            cmap='bwr', shading='gouraud',
            vmin = var_min, vmax = var_max,
        )

        # remove axis ticks 
        ax.set_xticks([])
        ax.set_yticks([])

        grid_row += 1
        if grid_row >= n_rows:
            grid_row = 0
            grid_col += 1

    # common colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cb = fig.colorbar(pcm, cax=cbar_ax)
    cb.set_label('Pressure (mbar)')
    cbar_ax.yaxis.set_label_position('left')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()

