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
from ..kma import ClusterProbabilities, ChangeProbabilities

# import constants
from .config import _faspect, _fsize, _fdpi


def GenOneYearDaily(yy=1981, month_ini=1):
    'returns one generic year in a list of datetimes. Daily resolution'

    dp1 = datetime(yy, month_ini, 1)
    dp2 = dp1+timedelta(days=365)

    return [dp1 + timedelta(days=i) for i in range((dp2-dp1).days)]

def Generate_PerpYear_Matrix(num_clusters, bmus_values, bmus_dates,
                             num_sim=1, month_ini=1):
    '''
    Calculates and returns matrix for stacked bar plotting

    bmus_dates - datetime.datetime (only works if daily resolution)
    bmus_values has to be 2D (time, nsim)
    '''

    # generate perpetual year list
    list_pyear = GenOneYearDaily(month_ini=month_ini)

    # generate aux arrays
    m_plot = np.zeros((num_clusters, len(list_pyear))) * np.nan
    bmus_dates_months = np.array([d.month for d in bmus_dates])
    bmus_dates_days = np.array([d.day for d in bmus_dates])

    # sort data
    for i, dpy in enumerate(list_pyear):
        _, s = np.where(
            [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
        )
        b = bmus_values[s,:]
        b = b.flatten()

        for j in range(num_clusters):
            _, bb = np.where([(j+1 == b)])  # j+1 starts at 1 bmus value!

            m_plot[j,i] = float(len(bb)/float(num_sim))/len(s)

    return m_plot


def axplot_PerpYear(ax, num_clusters, bmus_values, bmus_dates, num_sim, month_ini):
    'axes plot bmus perpetual year'

    # get cluster colors for stacked bar plot
    np_colors_int = GetClusterColors(num_clusters)

    # generate dateticks
    x_val = GenOneYearDaily(month_ini = month_ini)

    # generate plot matrix
    m_plot = Generate_PerpYear_Matrix(
        num_clusters, bmus_values, bmus_dates,
        num_sim = num_sim, month_ini = month_ini)

    # plot stacked bars
    bottom_val = np.zeros(m_plot[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot[r,:]
        ax.bar(
            x_val, row_val, bottom = bottom_val,
            width = 1, color = np.array([np_colors_int[r]])
        )

        # store bottom
        bottom_val += row_val

    # customize  axis
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b')

    ax.set_xlim(x_val[0], x_val[-1])
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.set_ylim(0, 1)
    ax.set_ylabel('')

def axplot_ChangeProbs(ax, change_probs, wt_colors,
                       ttl = '', vmin = 0, vmax = 1,
                       cmap = 'Blues', cbar_ttl = ''):
    'axes plot cluster change probabilities'

    num_clusters = change_probs.shape[0]

    # clsuter transition plot
    pc = ax.pcolor(
        change_probs,
        cmap=cmap, vmin=vmin, vmax=vmax,
    )

    # add colorbar
    cbar = plt.colorbar(pc, ax=ax)
    cbar.ax.tick_params(labelsize=8)
    if vmin != 0 or vmax !=1:
        cbar.set_ticks(np.linspace(vmin, vmax, 6))
    cbar.ax.set_ylabel(cbar_ttl, rotation=270, labelpad=20)

    # customize axes
    ax.set_xticks(np.arange(num_clusters)+0.5)
    ax.set_yticks(np.arange(num_clusters)+0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})

    # add custom color axis
    ccmap = mcolors.ListedColormap([tuple(r) for r in wt_colors])

    # custom color axis positions
    ax_pos = ax.get_position()
    cax_x_pos = [ax_pos.x0, ax_pos.y0-0.03, ax_pos.width, 0.02]
    cax_y_pos = [ax_pos.x0-0.03/_faspect, ax_pos.y0, 0.02/_faspect, ax_pos.height]

    # custom color axis X
    cax_x = ax.figure.add_axes(cax_x_pos)
    cbar_x = ColorbarBase(
        cax_x, cmap = ccmap, orientation='horizontal',
        norm = mcolors.Normalize(vmin=0, vmax=num_clusters),
    )
    cbar_x.set_ticks([])

    # custom color axis X
    cax_y= ax.figure.add_axes(cax_y_pos)
    cbar_y = ColorbarBase(
        cax_y, cmap = ccmap, orientation='vertical',
        norm = mcolors.Normalize(vmin=0, vmax=num_clusters),
    )
    cbar_y.set_ticks([])

def axplot_Transition(ax, probs_change_fit, probs_change_sim,
                      ttl='', color='black'):
    'axes plot for scatter historical - simulation transition matrix comparison'

    # clsuter transition fit vs. sim scatter plot
    pc = ax.scatter(
        probs_change_fit, probs_change_sim,
        color=color, s=5, zorder=2,
    )

    # get max value for axis lim
    axlim = np.ceil(10*np.max(probs_change_sim[:]))/10

    # customize axes
    ax.plot(([0,0],[1,1]), linestyle='dashed', linewidth=0.5, color='grey',
            zorder=1)
    ax.set_ylabel('Simulated', {'fontsize':8})
    ax.set_xlabel('Historical', {'fontsize':8})
    ax.grid(True, which='major', axis='both', linestyle='--', color='grey')
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.set_xlim([0, axlim])
    ax.set_ylim([0, axlim])
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})

def axplot_ClusterProbs(ax, cluster_probs_fit, cluster_probs_sim, wt_colors,
                       ttl=''):
    'axes plot for scatter historical - simulation cluster probs comparison'

    # clsuter transition plot
    for cf, cs, wc in zip(cluster_probs_fit, cluster_probs_sim, wt_colors):
        ax.plot(cf, cs, marker='.', markersize=8, color=wc, zorder=2)

    # get max value for axis lim
    axlim = np.ceil(10*np.max(cluster_probs_sim[:]))/10

    # customize axes
    ax.plot(([0,0],[1,1]), linestyle='dashed', linewidth=0.5, color='grey',
            zorder=1)
    ax.set_ylabel('Simulated', {'fontsize':8})
    ax.set_xlabel('Historical', {'fontsize':8})
    ax.grid(True, which='major', axis='both', linestyle='--', color='grey')
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.set_xlim([0, axlim])
    ax.set_ylim([0, axlim])
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})


def Plot_Compare_PerpYear(num_clusters,
                          bmus_values_sim, bmus_dates_sim,
                          bmus_values_hist, bmus_dates_hist,
                          n_sim = 1, month_ini = 1, p_export=None):
    '''
    Plot simulated - historical bmus comparison in a perpetual year

    bmus_dates requires 1 day resolution time
    bmus_values set min value has to be 1 (not 0)
    '''

    # check dates have 1 day time resolution 
    td_h = bmus_dates_hist[1] - bmus_dates_hist[0]
    td_s = bmus_dates_sim[1] - bmus_dates_sim[0]
    if td_h.days != 1 or td_s.days != 1:
        print('PerpetualYear bmus comparison skipped.')
        print('timedelta (days): Hist - {0}, Sim - {1})'.format(
            td_h.days, td_s.days))
        return


    # plot figure
    fig, (ax_hist, ax_sim) = plt.subplots(2,1, figsize=(_faspect*_fsize, _fsize))

    # historical perpetual year
    axplot_PerpYear(
        ax_hist, num_clusters, bmus_values_hist, bmus_dates_hist,
        num_sim = 1, month_ini = month_ini,
    )
    ax_hist.set_title('Historical')

    # simulated perpetual year
    axplot_PerpYear(
        ax_sim, num_clusters, bmus_values_sim, bmus_dates_sim,
        num_sim = n_sim, month_ini = month_ini,
    )
    ax_sim.set_title('Simulation')

    # add custom colorbar
    np_colors_int = GetClusterColors(num_clusters)
    ccmap = mcolors.ListedColormap(
        [tuple(r) for r in np_colors_int]
    )
    cax = fig.add_axes([0.92, 0.125, 0.025, 0.755])
    cbar = ColorbarBase(
        cax, cmap=ccmap,
        norm = mcolors.Normalize(vmin=0, vmax=num_clusters),
        ticks = np.arange(num_clusters) + 0.5,
    )
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticklabels(range(1, num_clusters + 1))

    # text
    fig.suptitle('Perpetual Year', fontweight='bold', fontsize=12)

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_Compare_Transitions(num_clusters, bmus_values_hist, bmus_values_sim,
                             p_export=None):

    # TODO: make compatible with bmus_values_sim size [n_xim x bmus]

    # get cluster colors
    wt_colors = GetClusterColors(num_clusters)
    wt_set = np.arange(num_clusters) + 1

    # cluster probabilities
    probs_c_fit = ClusterProbabilities(bmus_values_hist, wt_set)
    probs_c_sim = ClusterProbabilities(bmus_values_sim, wt_set)

    # cluster change probabilities
    _, chprobs_fit = ChangeProbabilities(bmus_values_hist, wt_set)
    _, chprobs_rnd = ChangeProbabilities(bmus_values_sim, wt_set)
    chprobs_dif = chprobs_rnd - chprobs_fit

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    # layout
    gs = gridspec.GridSpec(2, 6, wspace=0.60, hspace=0.35)
    ax_chpr_fit = plt.subplot(gs[0, :2])
    ax_chpr_rnd = plt.subplot(gs[0, 2:4])
    ax_chpr_dif = plt.subplot(gs[0, 4:])
    ax_scat_tra = plt.subplot(gs[1, 1:3])
    ax_scat_pbs = plt.subplot(gs[1, 3:5])

    # cluster change probs axes 
    axplot_ChangeProbs(ax_chpr_fit, chprobs_fit, wt_colors,
                       'Historical WT Transition Probabilities')
    axplot_ChangeProbs(ax_chpr_rnd, chprobs_rnd, wt_colors,
                       'Simulation WT Transition Probabilities')
    axplot_ChangeProbs(ax_chpr_dif, chprobs_dif, wt_colors,
                       'Change in WT Transition Probabilities',
                       vmin=-0.05, vmax=0.05, cmap='RdBu_r',
                       cbar_ttl='Simulation - Historical',
                      )

    # scatter plot transitions
    axplot_Transition(ax_scat_tra, chprobs_fit, chprobs_rnd,
                      'Sim. vs. Hist. WT Transition Probabilities'
                     )

    # scatter plot cluster probs
    axplot_ClusterProbs(ax_scat_pbs, probs_c_fit, probs_c_sim, wt_colors,
                       'Sim. vs. Hist. WT Probabilities',
                       )

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()





# TODO: following functions are not finished / tested

def Plot_PerpYear(bmus_values, bmus_dates, num_clusters, num_sim=1,
                  p_export=None):
    'Plots ARL bmus simulated in a perpetual_year stacked bar chart'

    # TODO: UPDATE


    # get cluster colors for stacked bar plot
    np_colors_int = GetClusterColors(num_clusters)

    # generate plot matrix
    m_plot = Generate_PerpYear_Matrix(
        num_clusters, bmus_values, bmus_dates, num_sim)

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_faspect*_fsize, _fsize))

    bottom_val = np.zeros(m_plot[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot[r,:]
        plt.bar(
            range(365), row_val, bottom=bottom_val,
            width=1, color = np.array([np_colors_int[r]])
               )

        # store bottom
        bottom_val += row_val

    # axis
    # TODO: REFINAR AXIS
    plt.xlim(1, 365)
    plt.ylim(0, 1)
    plt.xlabel('Perpetual year')
    plt.ylabel('')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

