#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy.interpolate import interp1d
import calendar
from datetime import datetime, timedelta

from lib.custom_dateutils import xds2datetime
from lib.util.operations import GetDivisors

# fig aspect and size
_faspect = (1+5**0.5)/2.0
_fsize = 7

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


def Plot_Weather_Types(xds_AWT, longitude, p_export=None):
    '''
    TODO DOC
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
        n_rows = l_div[len(l_div)/2]
        n_cols = n_clusters/n_rows

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.10, hspace=0.15)

    # TODO MEJORAR
    cs = ['b', 'g', 'r', 'c', 'm', 'k', 'y']

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

def Plot_3D_3PCs_WTs(xds_AWT, p_export=None):
    '''
    TODO DOC
    '''

    variance = xds_AWT.variance.values[:]
    bmus = xds_AWT.bmus.values[:]
    PCs = xds_AWT.PCs.values[:]
    order = xds_AWT.order.values[:]
    n_clusters = len(xds_AWT.n_clusters)

    PC1 = np.divide(PCs[:,0], np.sqrt(variance[0]))
    PC2 = np.divide(PCs[:,1], np.sqrt(variance[1]))
    PC3 = np.divide(PCs[:,2], np.sqrt(variance[2]))

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))
    ax = fig.add_subplot(111, projection='3d')

    # TODO MEJORAR
    cs = ['b', 'g', 'r', 'c', 'm', 'k', 'y']

    for ic in range(n_clusters):
        num = order[ic]
        ind =(np.where(bmus==num)[0][:])

        # scatter  plot
        ax.scatter(
            PC1[ind],PC2[ind],PC3[ind],
            c = cs[ic],
            label = 'WT#{0}'.format(ic+1)
        )

    ax.legend(loc='best')
    ax.set_xlabel('PC1',{'fontsize':10})
    ax.set_ylabel('PC2',{'fontsize':10})
    ax.set_zlabel('PC3',{'fontsize':10})

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

