#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
from math import sqrt
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy.interpolate import interp1d
import calendar
from datetime import datetime, timedelta

from lib.custom_dateutils import xds2datetime
from lib.util.operations import GetDivisors

# fig aspecti, size, export png dpi
_faspect = (1+5**0.5)/2.0
_fsize = 7
_fedpi = 128

# TODO: figure parameters
_fntsize_label = 8
_fntsize_legend = 8
_fntsize_title = 8
# etc

# Weather Type colors
wt_colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']

def Plot_KMArg_clusters_datamean(xds_datavar, bmus, p_export=None):
    '''
    TODO: documentar correctamente
    '''

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

def Plot_Weather_Types(xds_AWT, longitude, p_export=None):
    '''
    Plot Weather types

    xds_AWT: KMA output
    '''

    bmus = xds_AWT.bmus.values[:]
    order = xds_AWT.order.values[:]
    Km = xds_AWT.Km.values[:]
    n_clusters = len(xds_AWT.n_clusters)

    # Get number of rows and cols for gridplot 
    sqrt_clusters = sqrt(n_clusters)
    if sqrt_clusters.is_integer():
        n_rows = int(sqrt_clusters)
        n_cols = int(sqrt_clusters)
    else:
        l_div = GetDivisors(n_clusters)
        n_rows = l_div[len(l_div)//2]
        n_cols = n_clusters//n_rows

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.10, hspace=0.15)

    grid_row = 0
    grid_col = 0
    yt = np.arange(12)+1
    for ic in range(n_clusters):
        num = order[ic]
        var_AWT = Km[num,:]
        D = var_AWT.reshape(-1, len(longitude))
        nwts = len(np.where(bmus==num)[0][:])

        # grid plot
        ax = plt.subplot(gs[grid_row, grid_col])
        ax.pcolormesh(
            D,
            cmap='RdBu', shading='gouraud',
            vmin=-2, vmax=+2,
        )
        ax.set_title(
            'WT#{0} {1} years'.format(ic+1, nwts),
            {'fontsize':8, 'fontweight':'bold'}
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('month',{'fontsize':6})
        ax.set_xlabel('lon',{'fontsize':6})

        grid_col += 1
        if grid_col >= n_cols:
            grid_col = 0
            grid_row += 1

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_WTs_Dates(xds_AWT, p_export=None):
    '''
    Plot each Weather Type dates

    xds_AWT: KMA output
    '''

    bmus = xds_AWT.bmus.values[:]
    dates = xds_AWT.time.values[:]
    order = xds_AWT.order.values[:]
    n_clusters = len(xds_AWT.n_clusters)
    ys_str = np.array([str(d).split('-')[0] for d in dates])

    text_cycler = itertools.cycle(['bottom', 'top'])

    # plot figure
    fig, ax = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    for ic in range(n_clusters):
        num = order[ic]
        index = np.where(bmus==num)[0][:]

        ax.plot(
            dates[index], bmus[index],
            marker='+',markersize=7, linestyle='', color=wt_colors[ic]
        )
        va = 'bottom'
        for tx,ty,tt in zip(dates[index], bmus[index], ys_str[index]):
            ax.text(
                tx, ty, tt,
                {'fontsize':6},
                verticalalignment = next(text_cycler),
            )
        ax.set_ylabel('WT',{'fontsize':8})
        ax.set_xlabel('Year',{'fontsize':8})

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_3D_3PCs_WTs(d_wts, ttl='Weather Types PCs', p_export=None):
    '''
    Plots PC1 PC2 PC3 with 3D axis

    d_wts: dictionary. contains PCs (nx3) for each Weather Type
    '''

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))
    ax = fig.add_subplot(111, projection='3d')

    # plot each weather type
    wt_keys = sorted(d_wts.keys())
    for ic, k in enumerate(wt_keys):
        PC1 = d_wts[k][:,0]
        PC2 = d_wts[k][:,1]
        PC3 = d_wts[k][:,2]

        # scatter  plot
        ax.scatter(
            PC1, PC2, PC3,
            c = wt_colors[ic],
            label = k,
        )

    ax.legend(loc='best')
    ax.set_xlabel('PC1', {'fontsize':10})
    ax.set_ylabel('PC2', {'fontsize':10})
    ax.set_zlabel('PC3', {'fontsize':10})
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_Compare_WTs_hist(d_wts_fit, d_wts_rnd, p_export=None):
    '''
    Plots PC1 PC2 PC3 fit vs rnd histograms for each Weather Type
    Generates one figure for each Weather Type

    d_wts_*: dictionary. contains PCs (nx3) for each Weather Type
    (same keys required)
    '''

    # plot parameters  
    nb = 20

    # plot each weather type
    wt_keys = sorted(d_wts_fit.keys())
    for k in wt_keys:

        # get data
        PC1_fit = d_wts_fit[k][:,0]
        PC2_fit = d_wts_fit[k][:,1]
        PC3_fit = d_wts_fit[k][:,2]

        PC1_rnd = d_wts_rnd[k][:,0]
        PC2_rnd = d_wts_rnd[k][:,1]
        PC3_rnd = d_wts_rnd[k][:,2]

        # plot figure
        fig, axs = plt.subplots(
            ncols=2, nrows=3,
            figsize=(_faspect*_fsize, _fsize),
        )
        fig.suptitle(k, fontsize=10, fontweight='bold')

        # compare histograms
        axs[0,0].hist(PC1_fit, nb, density=True, label='PC1_fit')
        axs[0,1].hist(PC1_rnd, nb, density=True, label='PC1_rnd')
        axs[1,0].hist(PC2_fit, nb, density=True, label='PC2_fit')
        axs[1,1].hist(PC2_rnd, nb, density=True, label='PC2_rnd')
        axs[2,0].hist(PC3_fit, nb, density=True, label='PC3_fit')
        axs[2,1].hist(PC3_rnd, nb, density=True, label='PC3_rnd')

        # show / export
        if not p_export:
            plt.show()
        else:
            if not op.isdir(p_export):
                os.makedirs(p_export)
            p_tmp = op.join(p_export, 'hist_{0}.png'.format(k))
            fig.savefig(p_tmp, dpi=128)
            plt.close()

