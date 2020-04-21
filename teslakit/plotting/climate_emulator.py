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
from teslakit.plotting.custom_colors import GetClusterColors

# import constants
from teslakit.plotting.config import _faspect, _fsize, _fdpi


def axplot_bmus(ax, bmus, num_clusters, lab):
    'axes plot bmus series using colors'

    # colors
    wt_colors = GetClusterColors(num_clusters)

    # prepare nans as white 
    bmus = bmus.fillna(num_clusters+1)
    wt_colors = np.vstack([wt_colors, [0,0,0]])

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
    'axes plot variables series'

    # find start and end index
    nna = np.where(~np.isnan(vv))[0]
    x_0, x_1 = nna[0], nna[-1]

    # plot series
    ax.plot(nna, vv[nna], linestyle=ls, linewidth=0.5, color=lc)

    # customize axes
    ax.set_xlim(0, len(vv))

    ax.set_xticks([])
    ax.yaxis.tick_right()
    ax.tick_params(axis='both', which='both', labelsize=7)
    ax.set_ylabel(lab, rotation=0, fontweight='bold', labelpad=35)

def Plot_Complete(xds, show=True):
    '''
    Plot complete data variables inside xds

    xds - xarray.Dataset: AWT, MJO, DWT, Hs, Tp, Dir, SS, AT, MMSL, TWL
    '''
    # TODO: auto parameters inside xds

    # parameters
    n_cs_AWT = 6
    n_cs_MJO = 25
    n_cs_DWT = 42


    # plot figure
    fig, axs = plt.subplots(
        10, 1,
        figsize=(_faspect*_fsize, _fsize),
    )

    # AWT
    axplot_bmus(axs[0], xds.AWT, n_cs_AWT, 'AWT')

    # MJO
    axplot_bmus(axs[1], xds.MJO, n_cs_MJO, 'MJO')

    # DWT
    axplot_bmus(axs[2], xds.DWT, n_cs_DWT, 'DWT')

    # HS
    axplot_series(axs[3], xds.Hs, 'solid', 'cyan', 'Hs(m)')

    # TP
    axplot_series(axs[4], xds.Tp, 'solid', 'red', 'Tp(s)')

    # DIR
    axplot_series(axs[5], xds.Dir, 'dotted', 'black', 'Dir(º)')

    # TODO: historical storm surge?
    # SS
    if 'SS' in xds.variables:
        axplot_series(axs[6], xds.SS, 'solid', 'orange', 'SS(m)')
    else:
        axs[6].set_xticks([])
        axs[6].set_yticks([])
        axs[6].set_ylabel('SS(m)', rotation=0, fontweight='bold', labelpad=35)

    # AT
    axplot_series(axs[7], xds.AT, 'solid', 'purple', 'AT(m)')

    # MMSL
    axplot_series(axs[8], xds.MMSL, 'solid', 'green', 'MMSL(m)')

    # TWL
    axplot_series(axs[9], xds.TWL, 'solid', 'blue', 'TWL(m)')

    # suptitle
    fig.suptitle('Complete Data', fontweight='bold', fontsize=12)

    # reduce first 3 subplots
    gs = gridspec.GridSpec(10, 1, height_ratios = [1,1,1,3,3,3,3,3,3,3])
    for i in range(10):
        axs[i].set_position(gs[i].get_position(fig))

    # show and return figure
    if show: plt.show()
    return fig


def axplot_histcompare(ax, var_fit, var_sim, color='skyblue', n_bins=40,
                       label_1='Historical', label_2='Simulation', ttl=''):
    'axes plot histogram comparison between fit-sim variables'

    (_, bins, _) = ax.hist(var_fit, n_bins, weights=np.ones(len(var_fit)) / len(var_fit),
            alpha=0.9, color='white', ec='k', label = label_1)

    ax.hist(var_sim, bins=bins, weights=np.ones(len(var_sim)) / len(var_sim),
            alpha=0.7, color=color, ec='k', label = label_2)

    # customize axes
    ax.legend(prop={'size':8})
    ax.set_title(ttl)

def Plot_LevelVariables_Histograms(data_hist, data_sim, show=True):
    'Plots predicted mmsl data'

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))
    gs = gridspec.GridSpec(2, 2) #, wspace=0.0, hspace=0.0)

    # Level
    dh = data_hist['level'].values[:]; dh = dh[~np.isnan(dh)]
    ds = data_sim['level'].values[:]; ds = ds[~np.isnan(ds)]
    ax = plt.subplot(gs[0, 0])
    axplot_histcompare(ax, dh, ds, ttl='Level')

    # AT
    dh = data_hist['AT'].values[:]; dh = dh[~np.isnan(dh)]
    ds = data_sim['AT'].values[:]; ds = ds[~np.isnan(ds)]
    ax = plt.subplot(gs[0, 1])
    axplot_histcompare(ax, dh, ds, ttl='AT')

    # MMSL
    dh = data_hist['MMSL'].values[:]; dh = dh[~np.isnan(dh)]
    ds = data_sim['MMSL'].values[:]; ds = ds[~np.isnan(ds)]
    ax = plt.subplot(gs[1, 0])
    axplot_histcompare(ax, dh, ds, ttl='MMSL')

    # TWL
    dh = data_hist['TWL'].values[:]; dh = dh[~np.isnan(dh)]
    ds = data_sim['TWL'].values[:]; ds = ds[~np.isnan(ds)]
    ax = plt.subplot(gs[1, 1])
    axplot_histcompare(ax, dh, ds, ttl='TWL')

    # fig customization
    fig.suptitle(
        'Historical - Simulation Comparison',
        fontsize=14, fontweight = 'bold',
    )

    # show and return figure
    if show: plt.show()
    return fig

