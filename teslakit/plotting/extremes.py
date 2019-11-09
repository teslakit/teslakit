#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
from datetime import datetime, timedelta

# pip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d

# import constants
from .config import _faspect, _fsize, _fdpi

# TODO: REFACTOR CHROM Y SIGMA

def Plot_GEVParams(xda_gev_var, p_export=None):
    'Plot GEV params for a GEV parameter variable (sea_Hs, swell_1_Hs, ...)'

    name = xda_gev_var.name
    params = xda_gev_var.parameter.values[:]
    ss = int(np.sqrt(len(xda_gev_var.n_cluster)))  # this will fail if cant sqrt

    d_minmax = {
        'shape': (-0.2, 0.2),
    }

    # plot figure
    fig, axs = plt.subplots(2,2, figsize=(_faspect*_fsize, _fsize))
    axs = [i for sl in axs for i in sl]

    # empty last axis
    axs[3].axis('off')
    axs[3].text(0, 0.9, 'GEV parameters: {0}'.format(name),
                fontsize=20, fontweight='bold')

    for c, par in enumerate(params):
        ax = axs[c]

        par_values = xda_gev_var.sel(parameter=par).values[:]
        rr_pv = np.flipud(np.reshape(par_values,(ss,ss)).T)

        if par in d_minmax.keys():
            cl = d_minmax[par]
        else:
            cl = (np.min(par_values), np.max(par_values))

        ax.pcolor(rr_pv, cmap='coolwarm_r', clim=cl)

        # add grid and title
        ax.set_title(par, fontweight='bold')
        ax.grid(True, which='major', axis='both', linestyle='-', color='k')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_ChromosomesProbs(xds_chrom, p_export=None):
    'Plot chromosomes probabilities for each WT using chrom triangles'

    # TODO: colores poco fuertes
    # TODO: orden al colorcar los WTs ? mirar otras zonas de codigo
    # mirar tambien el orden de los WTs al generar chrom, si usamos sorted_bmus
    # desde un principio los WTs deberian respetar el orden de todo
    # TODO: lib/plotting/KMA.py Plot_KMArg_clusters_datamean para copiar estilo

    probs = xds_chrom.probs.values[:]
    chrom = xds_chrom.chrom.values[:]
    WTs = xds_chrom.WT
    ss = int(np.sqrt(len(WTs)))  # works for 36

    # triangle
    tsd = 10.0
    tsq = np.sqrt(tsd**2-(tsd/2)**2)
    tri_x = np.array([0, 0, tsq, 0])
    tri_y = np.array([0, tsd, tsd/2, 0])

    # chrom triangle positions (sea - swell_1 - swell_2)
    d_chrom_cntrs = {
        '100': (tsq, tsq/2),
        '010': (0, tsd),
        '001': (0, 0),
        '110': (tsq/2, tsd*3/4),
        '011': (0, tsd/2),
        '101': (tsq/2, tsd/4),
        '111': (tsq/3, tsd/2),
    }

    # max prob and size
    maxprob = np.nanmax(probs[:])
    maxsize = 1.5/maxprob

    # figure
    fig, axs = plt.subplots(ss, ss, figsize=(_faspect*_fsize*1.2, _fsize*1.2))
    axs = [i for sl in axs for i in sl]

    # plot each WT
    for c_wt, wt in enumerate(WTs):
        ax = axs[c_wt]

        # plot triangle
        ax.plot(tri_x, tri_y, 'k:', linewidth=1)

        # plot chromosomes prob circles
        for c_c, ch in enumerate(chrom):
            cntr = d_chrom_cntrs[''.join([str(int(x)) for x in ch])]
            if np.sum(ch) == 3:
                cl = [0,0,0]
            else:
                cl = ch
            ax.add_patch(
                plt.Circle(
                    cntr,
                    radius = maxsize * probs[c_wt, c_c],
                    edgecolor = cl,
                    facecolor = cl,
                    alpha = probs[c_wt, c_c] / maxprob,
                )
            )

        # waves family location text
        ax.text(tsq+0.75, tsq/2+0.5, 'SEA', va='center', ha='center', size=7)
        ax.text(0, tsd+0.5, 'SWELL_1', va='center', ha='center', size=7)
        ax.text(0, 0-0.5, 'SWELL_2', va='center', ha='center', size=7)

        # axis parameters
        ax.axis('equal')
        ax.axis('off')
        #ax.set_title('WT: {0}'.format(c_wt+1))

        fig.suptitle('Chromosomes Probabilities - WTs')


    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_SigmaCorrelation(xds_chrom, d_sigma, p_export=None):
    'Plot sigma correlation (Hs1-Hs2, Hs1-Tp1) for each WT using chrom triangles'

    # Get sigma correlation values for plot
    n_wts = len(d_sigma.keys())
    n_chs = len(d_sigma[list(d_sigma.keys())[0]].keys())
    ss = int(np.sqrt(n_wts))  # works for 36

    chrom = xds_chrom.chrom.values[:]

    sigma_plot = np.zeros((n_wts, n_chs)) * np.nan
    sigma_wtcrom = np.zeros((n_wts, n_chs)) * np.nan

    for c_wt, k_wt in enumerate(sorted(d_sigma.keys())):
        for c_ch, k_ch in enumerate(sorted(d_sigma[k_wt].keys())):

            corr = d_sigma[k_wt][k_ch]['corr']
            if corr.shape[0]==3:
                sigma_plot[c_wt, c_ch] = corr[0,1]  # one family. Hs - Tp corr
            elif corr.shape[0]==6:
                sigma_plot[c_wt, c_ch] = corr[0,3]  # two families. Hs1 - Hs2 corr

            sigma_wtcrom[c_wt, c_ch] = d_sigma[k_wt][k_ch]['wt_crom']

    # triangle
    tsd = 10.0
    tsq = np.sqrt(tsd**2-(tsd/2)**2)
    tri_x = np.array([0, 0, tsq, 0])
    tri_y = np.array([0, tsd, tsd/2, 0])

    # chrom triangle positions (sea - swell_1 - swell_2)
    d_chrom_cntrs = {
        '100': (tsq, tsq/2),
        '010': (0, tsd),
        '001': (0, 0),
        '110': (tsq/2, tsd*3/4),
        '011': (0, tsd/2),
        '101': (tsq/2, tsd/4),
        '111': (tsq/3, tsd/2),
    }
    # marker colors
    mk_pos = np.array([255, 0, 127]) / 255
    mk_neg = np.array([0, 127, 255]) / 255

    # max corr and size
    maxcorr = np.nanmax(sigma_plot[:])
    maxsize = 1.5/maxcorr

    # figure
    fig, axs = plt.subplots(ss, ss, figsize=(_faspect*_fsize*1.2, _fsize*1.2))
    axs = [i for sl in axs for i in sl]

    # plot each WT
    for c_wt, k_wt in enumerate(sorted(d_sigma.keys())):
        ax = axs[c_wt]

        # plot triangle
        ax.plot(tri_x, tri_y, 'k:', linewidth=1)

        # plot chromosomes prob circles
        for c_c, ch in enumerate(chrom):
            cntr = d_chrom_cntrs[''.join([str(int(x)) for x in ch])]

            # colors by sigma and wtcrom values
            if sigma_plot[c_wt, c_c] >= 0:
                mk_cl = mk_pos
            else:
                mk_cl = mk_neg

            if sigma_wtcrom[c_wt, c_c] == 0:
                e_cl = [0,0,0]
            else:
               e_cl = mk_cl

            # add circle    
            ax.add_patch(
                plt.Circle(
                    cntr,
                    radius = maxsize * np.absolute(sigma_plot[c_wt, c_c]),
                    edgecolor = e_cl,
                    facecolor = mk_cl,
                    alpha = np.absolute(sigma_plot[c_wt, c_c]) / maxcorr,
                )
            )

        # waves family location text
        ax.text(tsq+0.75, tsq/2+0.5, 'SEA', va='center', ha='center', size=7)
        ax.text(0, tsd+0.5, 'SWELL_1', va='center', ha='center', size=7)
        ax.text(0, 0-0.5, 'SWELL_2', va='center', ha='center', size=7)
        # TODO: textos HH HT etc

        # axis parameters
        ax.axis('equal')
        ax.axis('off')
        #ax.set_title('WT: {0}'.format(c_wt+1))

        fig.suptitle('Sigma Correlation - WTs')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_Schemaball(xds_data):
    # TODO
    return None

def Plot_ReturnPeriodValidation(xds_hist, xds_sim, var_name, p_export=None):
    'Plot Return Period historical - simulation validation'
    # TODO
    pass

    fig, axs = plt.subplots(111, figsize=(_faspect*_fsize, _fsize))

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

