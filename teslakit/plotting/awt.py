#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import itertools
import calendar
from datetime import datetime

# pip
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

# teslakit
from .custom_colors import colors_awt
from ..util.operations import GetBestRowsCols
from ..custom_dateutils import xds_reindex_daily as xr_daily
from ..custom_dateutils import xds_common_dates_daily as xcd_daily
from ..kma import ClusterProbabilities


# import constants
from .config import _faspect, _fsize, _fdpi

def axplot_AWT_2D(ax, var_2D, num_wts, id_wt, color_wt):
    'axes plot AWT variable (2D)'

    # plot 2D AWT
    ax.pcolormesh(
        var_2D,
        cmap='RdBu_r', shading='gouraud',
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

def axplot_PCs_2D(ax, PC1, PC2, d_wts, c_wts):
    'axes plot AWT PCs 1,2,3 (3D)'

    # calculate PC centroids
    pc1_wt = [np.mean(PC1[d_wts[i]]) for i in sorted(d_wts.keys())]
    pc2_wt = [np.mean(PC2[d_wts[i]]) for i in sorted(d_wts.keys())]

    # scatter  plot
    ax.scatter(
        PC1, PC2,
        c = 'silver',
        s = 3,
    )

    # WT centroids
    for x,y,c in zip(pc1_wt, pc2_wt, c_wts):
        ax.scatter(x, y, c=[c], s=10)

    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    ax.set_xticks([])
    ax.set_yticks([])

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

    ax.hist(pc_wt, nb, density=True, color=color_wt)

    # gridlines and axis properties
    ax.grid(True, which='both', axis='both', linestyle='--', color='grey')
    ax.set_xlim([-3,3])
    ax.set_yticklabels([])
    ax.tick_params(axis='x', which='major', labelsize=5)
    if ylab:
        ax.set_ylabel(ylab, {'fontweight':'bold'}, labelpad=-3)

def axplot_EOF_evolution(ax, years, EOF_evol):
    'axes plot EOFs evolution'

    # date axis locator
    yloc5 = mdates.YearLocator(5)
    yloc1 = mdates.YearLocator(1)
    yfmt = mdates.DateFormatter('%Y')

    # get years datetime
    ys_dt = np.array([datetime(y,1,1) for y in years])

    # plot EOF evolution 
    ax.plot(
        ys_dt, EOF_evol,
        linestyle='-', color='black',
    )

    # configure axis
    ax.set_xlim(ys_dt[0], ys_dt[-1])
    ax.xaxis.set_major_locator(yloc5)
    ax.xaxis.set_minor_locator(yloc1)
    ax.xaxis.set_major_formatter(yfmt)
    ax.grid(True, which='both', axis='x', linestyle='--', color='grey')
    ax.tick_params(axis='both', which='major', labelsize=8)

def axplot_EOF(ax, EOF_value, lon, ylbl, ttl):
    'axes plot EOFs evolution'

    # EOF pcolormesh 
    ax.pcolormesh(
        lon, range(12), np.transpose(EOF_value),
        cmap='RdBu_r', shading='gouraud',
        clim=2,
    )

    # axis and title
    ax.set_yticklabels(ylbl)
    ax.set_title(
        ttl,
        {'fontsize': 14, 'fontweight':'bold'}
    )
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.tick_params(axis='y', which='major', labelsize=10)

def axplot_DWT_Probs(ax, dwt_probs,
                     ttl = '', vmin = 0, vmax = 0.1,
                     cmap = 'Reds', caxis='black'):
    'axes plot DWT cluster probabilities'

    # clsuter transition plot
    ax.pcolor(
        np.flipud(dwt_probs),
        cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors='k',
    )

    # customize axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})

    plt.setp(ax.spines.values(), color=caxis, linewidth=4)
    plt.setp(
        [ax.get_xticklines(), ax.get_yticklines()],
        color=caxis,
    )


def Plot_AWT_Validation_Cluster(AWT_2D, AWT_num_wts, AWT_ID, AWT_dates,
                                AWT_bmus, AWT_PCs_fit, AWT_PCs_rnd, AWT_color,
                                p_export=None):

    from mpl_toolkits.mplot3d import Axes3D

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
    Plot Annual Weather Types
    '''

    bmus = xds_AWT.bmus.values[:]
    order = xds_AWT.order.values[:]
    Km = xds_AWT.Km.values[:]
    n_clusters = len(xds_AWT.n_clusters.values[:])

    # get number of rows and cols for gridplot 
    n_cols, n_rows = GetBestRowsCols(n_clusters)

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
    n_clusters = len(xds_AWT.n_clusters.values[:])

    # get cluster colors
    cs_awt = colors_awt()

    # plot figure
    fig, axs = plt.subplots(nrows=n_clusters, figsize=(_faspect*_fsize, _fsize))

    # each cluster has a figure
    for ic in range(n_clusters):

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
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_AWT_PCs_3D(d_PCs_fit, d_PCs_rnd, p_export=None):
    '''
    Plot Annual Weather Types PCs fit - rnd comparison (3D)
    '''

    from mpl_toolkits.mplot3d import Axes3D

    # get cluster colors
    cs_awt = colors_awt()

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize/1.66))
    gs = gridspec.GridSpec(1, 2, wspace=0.10, hspace=0.35)
    ax_fit = plt.subplot(gs[0, 0], projection='3d')
    ax_sim = plt.subplot(gs[0, 1], projection='3d')

    # Plot PCs (3D)
    axplot_PCs_3D_allWTs(ax_fit, d_PCs_fit,  cs_awt, ttl='PCs fit')
    axplot_PCs_3D_allWTs(ax_sim, d_PCs_rnd,  cs_awt, ttl='PCs sim')

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_AWT_PCs(xds_PCA, xds_KMA, n=3, p_export=None):
    '''
    Plot Annual Weather Types PCs using 2D axis
    '''

    # data
    PCs = xds_PCA.PCs.values[:]
    variance = xds_PCA.variance.values[:]
    bmus = xds_KMA.bmus_corrected.values[:]  # corrected bmus
    n_clusters = len(xds_KMA.n_clusters.values[:])

    # get cluster - bmus indexes
    d_wts = {}
    for i in range(n_clusters):
        d_wts[i] = np.where(bmus == i)[:]

    # get cluster colors
    cs_awt = colors_awt()

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _faspect*_fsize))
    gs = gridspec.GridSpec(n-1, n-1, wspace=0.0, hspace=0.0)

    for i in range(n):
        for j in range(i+1, n):

            # get PCs to plot
            PC1 = np.divide(PCs[:,i], np.sqrt(variance[i]))
            PC2 = np.divide(PCs[:,j], np.sqrt(variance[j]))

            # plot PCs (2D)
            ax = plt.subplot(gs[i, j-1])
            axplot_PCs_2D(ax, PC1, PC2, d_wts, cs_awt)

            # custom labels
            if i==0:
                ax.set_xlabel(
                    'PC {0}'.format(j+1),
                    {'fontsize':10, 'fontweight':'bold'}
                )
                ax.xaxis.set_label_position('top')
            if j==n-1:
                ax.set_ylabel(
                    'PC {0}'.format(i+1),
                    {'fontsize':10, 'fontweight':'bold'}
                )
                ax.yaxis.set_label_position('right')

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_EOFs_SST(xds_PCA, n_plot, p_export=None):
    '''
    Plot annual EOFs for 3D predictors

    xds_PCA:
        (n_components, n_components) PCs
        (n_components, n_features) EOFs
        (n_components, ) variance

        (n_lon, ) pred_lon: predictor longitude values

        attrs: y1, y2, m1, m2: PCA time parameters
        method: latitude averaged

    n_plot: number of EOFs plotted
    '''

    # PCA data
    variance = xds_PCA['variance'].values
    EOFs = np.transpose(xds_PCA['EOFs'].values)
    PCs = np.transpose(xds_PCA['PCs'].values)

    # PCA latavg metadata
    y1 = xds_PCA.attrs['y1']
    y2 = xds_PCA.attrs['y2']
    m1 = xds_PCA.attrs['m1']
    m2 = xds_PCA.attrs['m2']
    lon = xds_PCA['pred_lon'].values

    # mesh data
    len_x = len(lon)

    # time data
    years = range(y1, y2+1)
    l_months = [calendar.month_name[x] for x in range(1,13)]
    ylbl = l_months[m1-1:] + l_months[:m2]

    # percentage of variance each field explains
    n_percent = variance / np.sum(variance)

    for it in range(n_plot):

        # map of the spatial field
        spatial_fields = EOFs[:,it]*np.sqrt(variance[it])

        # reshape from vector to matrix with separated months
        C = np.reshape(
            spatial_fields[:len_x*12], (12, len_x)
        ).transpose()

        # plot figure
        fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

        # layout
        gs = gridspec.GridSpec(4, 4, wspace=0.10, hspace=0.2)
        ax_EOF = plt.subplot(gs[:3, :])
        ax_evol = plt.subplot(gs[3, :])

        # EOF pcolormesh
        ttl = 'EOF #{0}  ---  {1:.2f}%'.format(it+1, n_percent[it]*100)
        axplot_EOF(ax_EOF, C, lon, ylbl, ttl)

        # time series EOF evolution
        evol =  PCs[it,:]/np.sqrt(variance[it])
        axplot_EOF_evolution(ax_evol, years, evol)

        # show / export
        if not p_export:
            plt.show()

        else:
            if not op.isdir(p_export):
                os.makedirs(p_export)
            p_expi = op.join(p_export, 'EOFs_{0}.png'.format(it+1))
            fig.savefig(p_expi, dpi=_fdpi)
            plt.close()

def Plot_AWTs_DWTs_Probs(xds_AWT, ncs_AWT, xds_DWT, ncs_DWT, ttl='', p_export=None):
    '''
    Plot Annual Weather Types / Daily Weather Types probabilities

    both DWT and AWT bmus have to start at 0
    '''

    # reindex AWT to daily dates (year pad to days)
    xds_AWT = xr_daily(xds_AWT)

    # get common dates AWT-DWT
    d_comon = xcd_daily([xds_AWT, xds_DWT])
    xds_AWT = xds_AWT.sel(time=slice(d_comon[0], d_comon[-1]))
    xds_DWT = xds_DWT.sel(time=slice(d_comon[0], d_comon[-1]))

    # data for plotting
    awt_bmus = xds_AWT.bmus.values[:]
    awt_dats = xds_AWT.time.values[:]

    dwt_bmus = xds_DWT.bmus.values[:]
    dwt_dats = xds_DWT.time.values[:]

    # set of daily weather types
    dwt_set = np.arange(ncs_DWT)

    # dailt weather types matrix rows and cols
    n_rows, n_cols = GetBestRowsCols(ncs_DWT)

    # get cluster colors
    cs_awt = colors_awt()

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize/3))
    gs = gridspec.GridSpec(1, ncs_AWT, wspace=0.10, hspace=0.15)

    for ic in range(ncs_AWT):

        # select DWT bmus at current AWT indexes
        index_awt = np.where(awt_bmus==ic)[0][:]
        dwt_bmus_sel = dwt_bmus[index_awt]

        # get DWT cluster probabilities
        cps = ClusterProbabilities(dwt_bmus_sel, dwt_set)
        C_T = np.reshape(cps, (n_rows, n_cols))

        # plot axes
        ax_AWT = plt.subplot(gs[0, ic])
        axplot_DWT_Probs(
            ax_AWT, C_T,
            ttl = 'AWT {0}'.format(ic+1),
            cmap = 'Reds', caxis = cs_awt[ic],
        )
        ax_AWT.set_aspect('equal')

    # add fig title
    fig.suptitle(ttl, fontsize=14, fontweight='bold')

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

