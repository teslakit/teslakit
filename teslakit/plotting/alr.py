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
from scipy.interpolate import interp1d

# teslakit
from .util import MidpointNormalize
from .custom_colors import colors_awt, colors_mjo, colors_dwt, colors_interp
from ..custom_dateutils import xds2datetime as xds2dt
from ..kma import ClusterProbabilities, ChangeProbabilities

# import constants
from .config import _faspect, _fsize, _fdpi

def GetClusterColors(num_clusters):
    'Choose colors or Interpolate custom colormap to number of clusters'

    if num_clusters == 25:

        # mjo categ colors 
        np_colors_rgb = colors_mjo()

    elif num_clusters in [36, 42]:

        # dwt colors 
        np_colors_rgb = colors_dwt(num_clusters)

    else:
        # interpolate colors
        np_colors_rgb = colors_interp(num_clusters)

    return np_colors_rgb

def GenOneYearDaily(yy=1981, month_ini=1):
    'returns one generic year in a list of datetimes. Daily resolution'

    dp1 = datetime(yy, month_ini, 1)
    dp2 = dp1+timedelta(days=365)

    return [dp1 + timedelta(days=i) for i in range((dp2-dp1).days)]

def Plot_PValues(p_values, term_names, p_export=None):
    'Plot ARL/BMUS p-values'

    n_wts = p_values.shape[0]
    n_terms = p_values.shape[1]

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_faspect*_fsize, _fsize))

    c = ax.pcolor(p_values, cmap='coolwarm_r', clim=(0,1),
                  norm=MidpointNormalize(midpoint=0.1, vmin=0, vmax=1))
    #c.cmap.set_over('w')
    fig.colorbar(c, ax=ax)

    # Pval text
    for i in range(p_values.shape[1]):
        for j in range(p_values.shape[0]):
            v = p_values[j,i]
            if v<=0.1:
                ax.text(i+0.5, j+0.5, '{0:.2f}'.format(v),
                        va='center', ha='center', size=6)

    # axis
    ax.set_title('p-value', fontweight='bold')
    ax.set_ylabel('WT')

    ax.xaxis.tick_bottom()
    ax.set_xticks(np.arange(n_terms), minor=True)
    ax.set_xticks(np.arange(n_terms)+0.5, minor=False)
    ax.set_xticklabels(term_names, minor=False, rotation=90)

    ax.set_yticks(np.arange(n_wts), minor=True)
    ax.set_yticks(np.arange(n_wts)+0.5, minor=False)
    ax.set_yticklabels(np.arange(n_wts)+1, minor=False)

    # add grid
    ax.grid(True, which='minor', axis='both', linestyle='-', color='k')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_Params(params, term_names, p_export=None):
    'Plot ARL/BMUS params'

    n_wts = params.shape[0]
    n_terms = params.shape[1]

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_faspect*_fsize, _fsize))

    # text table and color
    c = ax.pcolor(params, cmap=plt.cm.bwr)
    for i in range(params.shape[1]):
        for j in range(params.shape[0]):
            v = params[j,i]
            ax.text(i+0.5, j+0.5, '{0:.1f}'.format(v),
                    va='center', ha='center', size=6)
    fig.colorbar(c, ax=ax)

    # axis
    ax.set_title('params', fontweight='bold')
    ax.set_ylabel('WT')

    ax.xaxis.tick_bottom()
    ax.set_xticks(np.arange(n_terms), minor=True)
    ax.set_xticks(np.arange(n_terms)+0.5, minor=False)
    ax.set_xticklabels(term_names, minor=False, rotation=90)

    ax.set_yticks(np.arange(n_wts), minor=True)
    ax.set_yticks(np.arange(n_wts)+0.5, minor=False)
    ax.set_yticklabels(np.arange(n_wts)+1, minor=False)

    # add grid
    ax.grid(True, which='minor', axis='both', linestyle='-', color='k')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Generate_PerpYear_Matrix(num_clusters, bmus_values, bmus_dates, num_sim=1,
                            month_ini=1):
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

def Generate_Covariate_Matrix(
    bmus_values, covar_values,
    bmus_dates, covar_dates,
    num_clusters, covar_rng, num_sim=1):
    'Calculates and returns matrix for stacked bar plotting'

    # generate aux arrays
    #bmus_years = [d.year for d in bmus_dates]
    m_plot = np.zeros((num_clusters, len(covar_rng)-1)) * np.nan

    for i in range(len(covar_rng)-1):

        # find years inside range
        _, s = np.where(
            [(covar_values >= covar_rng[i]) & (covar_values <= covar_rng[i+1])]
        )

        # TODO: usando alr_wrapper las fechas covar y bmus coinciden
        b = bmus_values[s,:]
        b = b.flatten()

        # TODO: mejorar, no usar los years y posicion. 
        # usar la fecha

        #ys = [covar_dates[x].year for x in s]
        # find data inside years found
        #sb = np.where(np.in1d(bmus_years, ys))[0]
        #b = bmus_values[sb,:]
        #b = b.flatten()

        for j in range(num_clusters):
            _, bb = np.where([(j+1 == b)])
            # TODO sb se utiliza para el test de laura
            #if len(sb) > 0:
                #m_plot[j,i] = float(len(bb)/float(num_sim))/len(sb)
            if len(s) > 0:
                m_plot[j,i] = float(len(bb)/float(num_sim))/len(s)
            else:
                m_plot[j,i] = 0

    return m_plot

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

def Plot_Covariate(bmus_values, covar_values,
                   bmus_dates, covar_dates,
                   num_clusters, name_covar,
                   num_sims=1,
                   p_export=None):
    'Plots ARL covariate related to bmus stacked bar chart'

    # get cluster colors for stacked bar plot
    np_colors_int = GetClusterColors(num_clusters)

    # generate common covar_rng
    covar_rng, interval = Generate_Covariate_rng(
        name_covar, covar_values)

    # generate plot matrix
    m_plot = Generate_Covariate_Matrix(
        bmus_values, covar_values,
        bmus_dates, covar_dates,
        num_clusters, covar_rng, num_sims)

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_faspect*_fsize, _fsize))
    x_val = covar_rng[:-1]

    bottom_val = np.zeros(m_plot[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot[r,:]
        plt.bar(
            x_val, row_val, bottom=bottom_val,
            width=interval, color = np.array([np_colors_int[r]])
               )

        # store bottom
        bottom_val += row_val

    # axis
    plt.xlim(np.min(x_val)-interval/2, np.max(x_val)+interval/2)
    plt.ylim(0, 1)
    plt.xlabel(name_covar)
    plt.ylabel('')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_Terms(terms_matrix, terms_dates, terms_names, p_export=None):
    'Plot terms used for ALR fitting'

    # number of terms
    n_sps = terms_matrix.shape[1]

    # custom fig size
    fsy = n_sps * 2

    # plot figure
    fig, ax_list = plt.subplots(
        n_sps, 1, sharex=True, figsize=(_faspect*_fsize, fsy)
    )

    x = terms_dates
    for i in range(n_sps):
        y = terms_matrix[:,i]
        n = terms_names[i]
        ax = ax_list[i]
        ax.plot(x, y, '.b')
        ax.set_title(n, loc='left', fontweight='bold', fontsize=10)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-1, 1)
        ax.grid(True, which='both')

        if n=='intercept':
            ax.set_ylim(0, 2)

    # date label
    fig.text(0.5, 0.04, 'date (y)', ha='center', fontweight='bold')
    fig.text(0.04, 0.5, 'value (-)', va='center', rotation='vertical',
             fontweight='bold')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_Compare_PerpYear(num_clusters,
                          bmus_values_sim, bmus_dates_sim,
                          bmus_values_hist, bmus_dates_hist,
                          n_sim = 1, month_ini=1, p_export=None):
    '''
    Plot simulated - historical bmus comparison in a perpetual year

    bmus_dates requires 1 day resolution time
    bmus_values set min value has to be 1 (not 0)
    '''

    # TODO: REFACTOR
    # TODO: months missing from x_axis?

    # check dates have 1 day time resolution 
    td_h = bmus_dates_hist[1] - bmus_dates_hist[0]
    td_s = bmus_dates_sim[1] - bmus_dates_sim[0]
    if td_h.days != 1 or td_s.days != 1:
        print('PerpetualYear bmus comparison skipped.')
        print('timedelta (days): Hist - {0}, Sim - {1})'.format(
            td_h.days, td_s.days))
        return

    # get cluster colors for stacked bar plot
    np_colors_int = GetClusterColors(num_clusters)

    # generate plot matrix
    m_plot_hist = Generate_PerpYear_Matrix(
        num_clusters, bmus_values_hist, bmus_dates_hist,
        num_sim=1, month_ini=month_ini)
    m_plot_sim = Generate_PerpYear_Matrix(
        num_clusters, bmus_values_sim, bmus_dates_sim,
        num_sim=n_sim, month_ini=month_ini)

    # plot figure
    fig, (ax_hist, ax_sim) = plt.subplots(2,1, figsize=(_faspect*_fsize, _fsize))

    # use dateticks
    x_val = GenOneYearDaily(month_ini=month_ini)

    # plot sim
    bottom_val = np.zeros(m_plot_sim[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot_sim[r,:]
        ax_sim.bar(
            x_val, row_val, bottom=bottom_val,
            width=1, color = np.array([np_colors_int[r]])
               )

        # store bottom
        bottom_val += row_val

    # plot hist
    bottom_val = np.zeros(m_plot_hist[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot_hist[r,:]
        ax_hist.bar(
            x_val, row_val, bottom=bottom_val,
            width=1, color = np_colors_int[r]
               )

        # store bottom
        bottom_val += row_val

    # axis
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b')
    years = mdates.YearLocator()
    yearsFmt = mdates.DateFormatter('')

    #ax_sim.set_xlim(1, 365)
    ax_sim.set_xlim(x_val[0], x_val[-1])
    ax_sim.xaxis.set_major_locator(years)
    ax_sim.xaxis.set_major_formatter(yearsFmt)
    ax_sim.xaxis.set_minor_locator(months)
    ax_sim.xaxis.set_minor_formatter(monthsFmt)
    ax_sim.set_ylim(0, 1)
    ax_sim.set_title('Simulation')
    ax_sim.set_ylabel('')

    #ax_hist.set_xlim(1, 365)
    ax_hist.set_xlim(x_val[0], x_val[-1])
    ax_hist.xaxis.set_major_locator(years)
    ax_hist.xaxis.set_major_formatter(yearsFmt)
    ax_hist.xaxis.set_minor_locator(months)
    ax_hist.xaxis.set_minor_formatter(monthsFmt)
    ax_hist.set_ylim(0, 1)
    ax_hist.set_title('Historical')
    ax_hist.set_ylabel('')

    # add custom colorbar
    ccmap = mcolors.ListedColormap(
        [tuple(r) for r in np_colors_int]
    )
    cax = fig.add_axes([0.92, 0.125, 0.025, 0.755])
    cbar = ColorbarBase(
        cax, cmap=ccmap,
        norm = mcolors.Normalize(vmin=0, vmax=num_clusters),
        ticks = range(1, num_clusters+1)
    )
    cbar.ax.tick_params(labelsize=8)

    # text
    fig.suptitle('Perpetual Year', fontweight='bold', fontsize=12)

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Generate_Covariate_rng(covar_name, cov_values):
    'Returns covar_rng and interval depending on covar_name'

    if covar_name.startswith('PC'):
        delta = 5
        n_rng = 7

        covar_rng = np.linspace(
            np.min(cov_values)-delta,
            np.max(cov_values)+delta,
            n_rng
        )

    elif covar_name.startswith('MJO'):
        delta = 0.5
        n_rng = 7

        covar_rng = np.linspace(
            np.min(cov_values)-delta,
            np.max(cov_values)+delta,
            n_rng
        )

    else:
        print('Cant plot {0}, missing rng params in plotting library'.format(
            name_covar
        ))
        return None, None

    # interval
    interval = covar_rng[1]-covar_rng[0]

    return covar_rng, interval

def Plot_Compare_Covariate(num_clusters,
                           bmus_values_sim, bmus_dates_sim,
                           bmus_values_hist, bmus_dates_hist,
                           cov_values_sim, cov_dates_sim,
                           cov_values_hist, cov_dates_hist,
                           name_covar,
                           n_sim = 1, p_export=None):
    'Plot simulated - historical bmus comparison related to covariate'

    # get cluster colors for stacked bar plot
    np_colors_int = GetClusterColors(num_clusters)

    # generate common covar_rng
    covar_rng, interval = Generate_Covariate_rng(
        name_covar, np.concatenate((cov_values_sim, cov_values_hist)))

    # generate plot matrix
    m_plot_sim = Generate_Covariate_Matrix(
        bmus_values_sim, cov_values_sim,
        bmus_dates_sim, cov_dates_sim,
        num_clusters, covar_rng, n_sim)

    m_plot_hist = Generate_Covariate_Matrix(
        bmus_values_hist, cov_values_hist,
        bmus_dates_hist, cov_dates_hist,
        num_clusters, covar_rng, 1)

    # plot figure
    fig, (ax_hist, ax_sim) = plt.subplots(2,1, figsize=(_faspect*_fsize, _fsize))
    x_val = covar_rng[:-1]

    # sim
    bottom_val = np.zeros(m_plot_sim[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot_sim[r,:]
        ax_sim.bar(
            x_val, row_val, bottom=bottom_val,
            width=interval, color = np.array([np_colors_int[r]])
               )

        # store bottom
        bottom_val += row_val

    # hist
    bottom_val = np.zeros(m_plot_hist[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot_hist[r,:]
        ax_hist.bar(
            x_val, row_val, bottom=bottom_val,
            width=interval, color = np_colors_int[r]
               )

        # store bottom
        bottom_val += row_val

    # axis
    # MEJORAR Y METER EL NOMBRE DE LA COVARIATE
    ax_sim.set_xlim(np.min(x_val)-interval/2, np.max(x_val)+interval/2)
    ax_sim.set_ylim(0, 1)
    ax_sim.set_title('Simulation')
    ax_sim.set_ylabel('')
    ax_sim.set_xlabel('')

    ax_hist.set_xlim(np.min(x_val)-interval/2, np.max(x_val)+interval/2)
    ax_hist.set_ylim(0, 1)
    ax_hist.set_title('Historical')
    ax_hist.set_ylabel('')
    ax_sim.set_xlabel('')

    fig.suptitle(name_covar, fontweight='bold', fontsize=12)

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

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

def Plot_Validation(num_clusters, bmus_values_hist, bmus_values_sim,
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
    gs = gridspec.GridSpec(2, 3, wspace=0.10, hspace=0.35)
    ax_chpr_fit = plt.subplot(gs[0, 0])
    ax_chpr_rnd = plt.subplot(gs[0, 1])
    ax_chpr_dif = plt.subplot(gs[0, 2])
    ax_scat_tra = plt.subplot(gs[1, 0])
    ax_scat_pbs = plt.subplot(gs[1, 2])

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
                      'Simualted vs. Historical WT Transition Probabilities'
                     )

    # scatter plot cluster probs
    axplot_ClusterProbs(ax_scat_pbs, probs_c_fit, probs_c_sim, wt_colors,
                       'Simulated vs. Historical WT Probabilities',
                       )

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

