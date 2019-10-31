#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

# pip
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# teslakit
from ..util.operations import GetBestRowsCols
from .custom_colors import GetFamsColors

# import constants
from .config import _faspect, _fsize, _fdpi


def axplot_distplot(ax, vars_values, vars_colors, n_bins, wt_num, xlims):
    'axes plot seaborn distplot variable at families'

    for vv, vc in zip(vars_values, vars_colors):
        sns.distplot(vv, bins=n_bins, color=tuple(vc), ax=ax);

    # wt text
    ax.text(0.87, 0.85, wt_num, transform=ax.transAxes, fontweight='bold')

    # customize axes
    ax.set_xlim(xlims)
    ax.set_xticks([])
    ax.set_yticks([])

def axplot_polarhist(ax, vars_values, vars_colors, n_bins, wt_num):
    'axes plot seaborn polar hist dir at families'

    for vv, vc in zip(vars_values, vars_colors):
        plt.hist(
            np.deg2rad(vv),
            range = [0, np.deg2rad(360)],
            bins = n_bins, color = vc,
            histtype='stepfilled', alpha = 0.5,
        )

    # wt text
    ax.text(0.87, 0.85, wt_num, transform=ax.transAxes, fontweight='bold')

    # customize axes
    ax.set_facecolor('whitesmoke')
    ax.set_xticks([])
    ax.set_yticks([])


def Plot_Waves_DWTs(xds_wvs_fams_sel, bmus, n_clusters, p_export=None):
    '''
    Plot waves families by DWT

    wvs_fams (waves families):
        xarray.Dataset (time,), fam1_Hs, fam1_Tp, fam1_Dir, ...
        {any number of families}

    xds_DWTs - ESTELA predictor KMA
        xarray.Dataset (time,), bmus, ...
    '''

    # plot_parameters
    n_bins = 35

    # get families names and colors
    n_fams = [vn.replace('_Hs','') for vn in xds_wvs_fams_sel.keys() if '_Hs' in vn]
    fams_colors = GetFamsColors(len(n_fams))

    # get number of rows and cols for gridplot 
    n_rows, n_cols = GetBestRowsCols(n_clusters)

    # Hs and Tp
    for wv in ['Hs', 'Tp']:

        # get common xlims for histogram
        allvals = np.concatenate(
            [xds_wvs_fams_sel['{0}_{1}'.format(fn, wv)].values[:] for fn in n_fams]
        )
        av_min, av_max = np.nanmin(allvals), np.nanmax(allvals)
        xlims = [math.floor(av_min), av_max]

        # figure
        fig = plt.figure(figsize=(_faspect*_fsize, _fsize))
        gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.0)
        gr, gc = 0, 0
        for ic in range(n_clusters):

            # data mean at clusters
            pc = np.where(bmus==ic)[0][:]
            xds_wvs_c = xds_wvs_fams_sel.isel(time=pc)
            vrs = [xds_wvs_c['{0}_{1}'.format(fn, wv)].values[:] for fn in n_fams]

            # axes plot
            ax = plt.subplot(gs[gr, gc])
            axplot_distplot(
                ax, vrs,
                fams_colors, n_bins,
                wt_num = ic+1,
                xlims=xlims,
            )

            # fig legend
            if gc == 0 and gr == 0:
                plt.legend(
                    title = 'Families',
                    labels = n_fams,
                    bbox_to_anchor=(1, 1),
                    bbox_transform=fig.transFigure,
                )

            # counter
            gc += 1
            if gc >= n_cols:
                gc = 0
                gr += 1

        fig.suptitle(
            '{0} Distributions: {1}'.format(wv, ', '.join(n_fams)),
            fontsize=14, fontweight = 'bold')

        # show / export
        if not p_export:
            plt.show()
        else:
            nme = 'wvs_fams_{0}.png'.format(wv)
            p_e = op.join(p_export, nme)
            fig.savefig(p_e, dpi=_fdpi)
            plt.close()

    # Dir    
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.1)
    gr, gc = 0, 0
    for ic in range(n_clusters):

        # data mean at clusters
        pc = np.where(bmus==ic)[0][:]
        xds_wvs_c = xds_wvs_fams_sel.isel(time=pc)
        vrs = [xds_wvs_c['{0}_Dir'.format(fn)].values[:] for fn in n_fams]

        # axes plot
        ax = plt.subplot(
            gs[gr, gc],
            projection='polar',
            theta_direction = -1, theta_offset = np.pi/2,
        )
        axplot_polarhist(
            ax, vrs,
            fams_colors, n_bins,
            wt_num = ic+1,
        )

        # fig legend
        if gc == n_cols-1 and gr==0:
            plt.legend(
                title = 'Families',
                labels = n_fams,
                bbox_to_anchor=(1, 1),
                bbox_transform=fig.transFigure,
            )

        # counter
        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

    fig.suptitle(
        '{0} Distributions: {1}'.format('Dir', ', '.join(n_fams)),
        fontsize=14, fontweight='bold')

    # show / export
    if not p_export:
        plt.show()
    else:
        nme = 'wvs_fams_dir_DWTs.png'
        p_e = op.join(p_export, nme)
        fig.savefig(p_e, dpi=_fdpi)
        plt.close()

