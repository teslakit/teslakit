#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
from datetime import datetime, timedelta

# pip
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

# teslakit
from .custom_colors import GetClusterColors

# import constants
from .config import _faspect, _fsize, _fdpi


def axplot_bmus(ax, bmus, num_clusters, lab):
    'axes plot bmus series using colors'

    # colors
    wt_colors = GetClusterColors(num_clusters)

    # bmus colors
    vp = wt_colors[bmus.astype(int)-1]

    # listed colormap
    ccmap = mcolors.ListedColormap(vp)

    # color series
    cbar_x = ColorbarBase(
        ax, cmap = ccmap, orientation='horizontal',
        norm = mcolors.Normalize(vmin=0, vmax=num_clusters),
    )
    cbar_x.set_ticks([])

    # customize axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(lab, rotation=0, fontweight='bold', labelpad=35)

def axplot_series(ax, vv, ls, lc, lab):
    'axes plot bmus series using colors'

    # plot series
    ax.plot(vv.time, vv, linestyle=ls, linewidth=0.5, color=lc)

    # customize axes
    ax.set_xlim(vv.time.values[0], vv.time.values[-1])
    ax.set_xticks([])
    ax.yaxis.tick_right()
    ax.tick_params(axis='both', which='both', labelsize=7)
    ax.set_ylabel(lab, rotation=0, fontweight='bold', labelpad=35)


def Plot_Simulation(xds_out_h, show=True):
    '''
    Plot all simulated output variables
    '''
    # TODO: auto parameters inside xds_out_h

    # parameters
    n_cs_AWT = 6
    n_cs_MJO = 25
    n_cs_DWT = 42


    # plot figure
    fig, axs = plt.subplots(10, 1, figsize=(_faspect*_fsize, _fsize))

    # AWT
    axplot_bmus(axs[0], xds_out_h.AWT, n_cs_AWT, 'AWT')

    # MJO
    axplot_bmus(axs[1], xds_out_h.MJO, n_cs_MJO, 'MJO')

    # DWT
    axplot_bmus(axs[2], xds_out_h.DWT, n_cs_DWT, 'DWT')

    # HS
    axplot_series(axs[3], xds_out_h.Hs, 'solid', 'cyan', 'Hs(m)')

    # TP
    axplot_series(axs[4], xds_out_h.Tp, 'solid', 'red', 'Tp(s)')

    # DIR
    axplot_series(axs[5], xds_out_h.Dir, 'dotted', 'black', 'Dir(º)')

    # SS
    axplot_series(axs[6], xds_out_h.SS, 'solid', 'orange', 'SS(m)')

    # AT
    axplot_series(axs[7], xds_out_h.AT, 'solid', 'purple', 'AT(m)')

    # MMSL
    axplot_series(axs[8], xds_out_h.MMSL, 'solid', 'green', 'MMSL(m)')

    # TWL
    axplot_series(axs[9], xds_out_h.TWL, 'solid', 'blue', 'TWL(m)')

    # suptitle
    fig.suptitle('Simulation', fontweight='bold', fontsize=12)

    # reduce first 3 subplots
    gs = gridspec.GridSpec(10, 1, height_ratios = [1,1,1,3,3,3,3,3,3,3])
    for i in range(10):
        axs[i].set_position(gs[i].get_position(fig))

    # show and return figure
    if show: plt.show()
    return fig

