#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy.interpolate import interp1d
import calendar
from datetime import datetime, timedelta

from lib.custom_dateutils import xds2datetime
from lib.util.operations import GetDivisors


def Plot_KMArg_clusters_datamean(xds_datavar, bmus, p_export=None):
    '''
    TODO
    '''

    # get some data
    clusters = sorted(set(bmus))
    n_clusters = len(clusters)
    var_max = np.max(xds_datavar.values)
    var_min = np.min(xds_datavar.values)

    # prepare figure
    fig = plt.figure(figsize=(16,9))

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

        # grid plot
        ax = plt.subplot(gs[grid_row, grid_col])
        ax.pcolormesh(
            np.flipud(mean_cluster.values),
            cmap='RdBu', shading='gouraud',
        )
        # TODO: plot with same climits
        ax.set_xticks([])
        ax.set_yticks([])

        grid_row += 1
        if grid_row >= n_rows:
            grid_row = 0
            grid_col += 1

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()

