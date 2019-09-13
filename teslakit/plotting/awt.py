#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import itertools

# pip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

# teslakit
from .custom_colors import colors_awt
from ..util.operations import GetDivisors

# import constants
from .config import _faspect, _fsize, _fdpi

def axplot_AWT_2D(ax, var_2D, num_wts, id_wt, color_wt):
    'axes plot AWT variable (2D)'

    # plot 2D AWT
    ax.pcolormesh(
        var_2D,
        cmap='RdBu', shading='gouraud',
        vmin=-1.5, vmax=+1.5,
    )

    # title and axis labels/ticks
    ax.set_title(
        'WT #{0} --- {1} years'.format(id_wt, num_wts),
        {'fontsize': 14, 'fontweight':'bold'}
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('month', {'fontsize':8})
    ax.set_xlabel('lon', {'fontsize':8})

    # set WT color on axis frame
    plt.setp(ax.spines.values(), color=color_wt, linewidth=4)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=color_wt)

def axplot_AWT_years(ax, dates_wt, bmus_wt, color_wt, xticks_clean=False,
                     ylab=None):
    'axes plot AWT dates'

    # date axis locator
    yloc5 = mdates.YearLocator(5)
    yloc1 = mdates.YearLocator(1)
    yfmt = mdates.DateFormatter('%Y')

    # get years string
    ys_str = np.array([str(d).split('-')[0] for d in dates_wt])

    # use a text bottom - top cycler
    text_cycler_va = itertools.cycle(['bottom', 'top'])
    text_cycler_ha = itertools.cycle(['left', 'right'])

    # plot AWT dates and bmus
    ax.plot(
        dates_wt, bmus_wt,
        marker='+',markersize=9, linestyle='', color=color_wt,
    )
    va = 'bottom'
    for tx,ty,tt in zip(dates_wt, bmus_wt, ys_str):
        ax.text(
            tx, ty, tt,
            {'fontsize':8},
            verticalalignment = next(text_cycler_va),
            horizontalalignment = next(text_cycler_ha),
            rotation=45,
        )

    # configure axis
    ax.set_yticks([])
    ax.xaxis.set_major_locator(yloc5)
    ax.xaxis.set_minor_locator(yloc1)
    ax.xaxis.set_major_formatter(yfmt)
    ax.grid(True, which='both', axis='x', linestyle='--', color='grey')
    ax.tick_params(axis='x', which='major', labelsize=8)

    # optional parameters
    if xticks_clean:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Year', {'fontsize':8})

    if ylab: ax.set_ylabel(ylab)

def axplot_PCs_3D(ax, pcs_wt, color_wt, ttl='PCs'):
    'axes plot AWT PCs 1,2,3 (3D)'

    PC1 = pcs_wt[:,0]
    PC2 = pcs_wt[:,1]
    PC3 = pcs_wt[:,2]

    # scatter  plot
    ax.scatter(
        PC1, PC2, PC3,
        c = [color_wt],
        s = 2,
    )

    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    ax.set_zlim([-3,3])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title(ttl, {'fontsize':8, 'fontweight':'bold'})

def axplot_PCs_3D_allWTs(ax, d_PCs, wt_colors, ttl='PCs'):
    'axes plot AWT PCs 1,2,3 (3D)'

    # plot each weather type
    wt_keys = sorted(d_PCs.keys())
    for ic, k in enumerate(wt_keys):
        PC1 = d_PCs[k][:,0]
        PC2 = d_PCs[k][:,1]
        PC3 = d_PCs[k][:,2]

        # scatter  plot
        ax.scatter(
            PC1, PC2, PC3,
            c = [wt_colors[ic]],
            label = k,
            s = 3,
        )

    ax.set_xlabel('PC1', {'fontsize':10})
    ax.set_ylabel('PC2', {'fontsize':10})
    ax.set_zlabel('PC3', {'fontsize':10})
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    ax.set_zlim([-3,3])
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})

def axplot_PC_hist(ax, pc_wt, color_wt, nb=30, ylab=None):
    'axes plot AWT singular PC histogram'

    # TODO: remove color_wt?
    #color_wt='dimgray'
    ax.hist(pc_wt, nb, density=True, color=color_wt)

    # gridlines and axis properties
    ax.grid(True, which='both', axis='both', linestyle='--', color='grey')
    ax.set_xlim([-3,3])
    ax.set_yticklabels([])
    ax.tick_params(axis='x', which='major', labelsize=5)
    if ylab:
        ax.set_ylabel(ylab, {'fontweight':'bold'}, labelpad=-3)


def Plot_AWT_Validation_Cluster(AWT_2D, AWT_num_wts, AWT_ID, AWT_dates,
                                AWT_bmus, AWT_PCs_fit, AWT_PCs_rnd, AWT_color, p_export=None):

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    # layout
    gs = gridspec.GridSpec(4, 4, wspace=0.10, hspace=0.15)
    ax_AWT_2D = plt.subplot(gs[:2, :2])
    ax_PCs3D_fit = plt.subplot(gs[2, 0], projection='3d')
    ax_PCs3D_rnd = plt.subplot(gs[2, 1], projection='3d')
    ax_AWT_y = plt.subplot(gs[3, :])
    ax_PC1_hst_fit = plt.subplot(gs[0, 2])
    ax_PC1_hst_rnd = plt.subplot(gs[0, 3])
    ax_PC2_hst_fit = plt.subplot(gs[1, 2])
    ax_PC2_hst_rnd = plt.subplot(gs[1, 3])
    ax_PC3_hst_fit = plt.subplot(gs[2, 2])
    ax_PC3_hst_rnd = plt.subplot(gs[2, 3])

    # plot AWT 2D
    axplot_AWT_2D(ax_AWT_2D, AWT_2D, AWT_num_wts, AWT_ID, AWT_color)

    # plot AWT years
    axplot_AWT_years(ax_AWT_y, AWT_dates, AWT_bmus, AWT_color)

    # compare PCs fit - sim with 3D plot
    axplot_PCs_3D(ax_PCs3D_fit, AWT_PCs_fit,  AWT_color, ttl='PCs fit')
    axplot_PCs_3D(ax_PCs3D_rnd, AWT_PCs_rnd,  AWT_color, ttl='PCs sim')

    # compare PC1 histograms
    axplot_PC_hist(ax_PC1_hst_fit, AWT_PCs_fit[:,0], AWT_color)
    axplot_PC_hist(ax_PC1_hst_rnd, AWT_PCs_rnd[:,0], AWT_color, ylab='PC1')

    axplot_PC_hist(ax_PC2_hst_fit, AWT_PCs_fit[:,1], AWT_color)
    axplot_PC_hist(ax_PC2_hst_rnd, AWT_PCs_rnd[:,1], AWT_color, ylab='PC2')

    axplot_PC_hist(ax_PC3_hst_fit, AWT_PCs_fit[:,2], AWT_color)
    axplot_PC_hist(ax_PC3_hst_rnd, AWT_PCs_rnd[:,2], AWT_color, ylab='PC3')

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_AWT_Validation(xds_AWT, lon, d_PCs_fit, d_PCs_rnd, p_export=None):
    '''
    Plot Annual Weather Types Validation

    xds_AWT               - KMA output
    lon                   - predictor longitude
    d_PCs_fit, d_PCs_rnd 1,2,3
    '''

    # TODO: activate p_export

    # get data
    bmus = xds_AWT.bmus_corrected.values[:]  # corrected bmus
    dates = xds_AWT.time.values[:]
    order = xds_AWT.order.values[:]  # clusters order
    Km = xds_AWT.Km.values[:]
    n_clusters = len(xds_AWT.n_clusters.values[:])

    # get cluster colors
    cs_awt = colors_awt()

    # each cluster has a figure
    for ic in range(n_clusters):

        # get cluster data
        num = order[ic]

        id_AWT = ic + 1           # cluster ID
        index = np.where(bmus==ic)[0][:]
        dates_AWT = dates[index]  # cluster dates
        bmus_AWT = bmus[index]    # cluster bmus
        var_AWT = Km[num,:]
        var_AWT_2D = var_AWT.reshape(-1, len(lon))
        num_WTs = len(index)      # number of cluster ocurrences
        clr = cs_awt[ic]          # cluster color
        PCs_fit = d_PCs_fit['{0}'.format(id_AWT)]
        PCs_rnd = d_PCs_rnd['{0}'.format(id_AWT)]

        # plot cluster figure
        Plot_AWT_Validation_Cluster(
            var_AWT_2D, num_WTs, id_AWT,
            dates_AWT, bmus_AWT,
            PCs_fit, PCs_rnd,
            clr)

def Plot_AWTs(xds_AWT, lon, p_export=None):
    '''
    Plot Annual Weather types
    '''

    bmus = xds_AWT.bmus.values[:]
    order = xds_AWT.order.values[:]
    Km = xds_AWT.Km.values[:]
    n_clusters = len(xds_AWT.n_clusters.values[:])

    ## Get number of rows and cols for gridplot 
    #sqrt_clusters = sqrt(n_clusters)
    #if sqrt_clusters.is_integer():
    #    n_rows = int(sqrt_clusters)
    #    n_cols = int(sqrt_clusters)
    #else:
    #    l_div = GetDivisors(n_clusters)
    #    n_rows = l_div[len(l_div)//2]
    #    n_cols = n_clusters//n_rows

    # TODO 6 AWTs
    n_rows = 2
    n_cols = 3

    # get cluster colors
    cs_awt = colors_awt()

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.10, hspace=0.15)
    gr = 0
    gc = 0

    for ic in range(n_clusters):
        num = order[ic]

        id_AWT = ic + 1           # cluster ID
        index = np.where(bmus==num)[0][:]
        var_AWT = Km[num,:]
        var_AWT_2D = var_AWT.reshape(-1, len(lon))
        num_WTs = len(index)
        clr = cs_awt[ic]          # cluster color

        # AWT var 2D 
        ax = plt.subplot(gs[gr, gc])
        axplot_AWT_2D(ax, var_AWT_2D, num_WTs, id_AWT, clr)

        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_AWTs_Dates(xds_AWT, p_export=None):
    '''
    Plot Annual Weather Types dates

    xds_AWT: KMA output
    '''

    # get data
    bmus = xds_AWT.bmus_corrected.values[:]  # corrected bmus
    dates = xds_AWT.time.values[:]
    order = xds_AWT.order.values[:]
    n_clusters = len(xds_AWT.n_clusters.values[:])

    # get cluster colors
    cs_awt = colors_awt()

    # plot figure
    fig, axs = plt.subplots(nrows=n_clusters, figsize=(_faspect*_fsize, _fsize))

    # each cluster has a figure
    for ic in range(n_clusters):
        num = order[ic]

        id_AWT = ic + 1           # cluster ID
        index = np.where(bmus==ic)[0][:]
        dates_AWT = dates[index]  # cluster dates
        bmus_AWT = bmus[index]    # cluster bmus
        clr = cs_awt[ic]          # cluster color

        ylabel = "WT #{0}".format(id_AWT)

        xaxis_clean=True
        if ic == n_clusters-1:
            xaxis_clean=False

        # axs plot
        axplot_AWT_years(axs[ic], dates_AWT, bmus_AWT, clr, xaxis_clean,
                         ylabel)

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_AWT_PCs_3D(d_PCs_fit, d_PCs_rnd, p_export=None):
    '''
    Plot Annual Weather Types PCs fit - rnd comparison (3D)
    '''

    # get cluster colors
    cs_awt = colors_awt()

    # plot figure
    fig, axs = plt.subplots(
        ncols=2, figsize=(_faspect*_fsize, _fsize),
        subplot_kw={'projection':'3d'})

    axplot_PCs_3D_allWTs(axs[0], d_PCs_fit,  cs_awt, ttl='PCs fit')
    axplot_PCs_3D_allWTs(axs[1], d_PCs_rnd,  cs_awt, ttl='PCs sim')

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()
