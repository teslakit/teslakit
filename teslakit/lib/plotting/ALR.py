#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime, timedelta

from lib.plotting.util import MidpointNormalize
from lib.custom_dateutils import xds2datetime as xds2dt

def GetClusterColors(num_clusters):
    'Interpolate custom colormap to number of clusters'
    # TODO: mejorarlo con los cmaps de laura u otros metodos

    l_colors_dwt = [
        (1.0000, 0.1344, 0.0021),
        (1.0000, 0.2669, 0.0022),
        (1.0000, 0.5317, 0.0024),
        (1.0000, 0.6641, 0.0025),
        (1.0000, 0.9287, 0.0028),
        (0.9430, 1.0000, 0.0029),
        (0.6785, 1.0000, 0.0031),
        (0.5463, 1.0000, 0.0032),
        (0.2821, 1.0000, 0.0035),
        (0.1500, 1.0000, 0.0036),
        (0.0038, 1.0000, 0.1217),
        (0.0039, 1.0000, 0.2539),
        (0.0039, 1.0000, 0.4901),
        (0.0039, 1.0000, 0.6082),
        (0.0039, 1.0000, 0.8444),
        (0.0039, 1.0000, 0.9625),
        (0.0039, 0.8052, 1.0000),
        (0.0039, 0.6872, 1.0000),
        (0.0040, 0.4510, 1.0000),
        (0.0040, 0.3329, 1.0000),
        (0.0040, 0.0967, 1.0000),
        (0.1474, 0.0040, 1.0000),
        (0.2655, 0.0040, 1.0000),
        (0.5017, 0.0040, 1.0000),
        (0.6198, 0.0040, 1.0000),
        (0.7965, 0.0040, 1.0000),
        (0.8848, 0.0040, 1.0000),
        (1.0000, 0.0040, 0.9424),
        (1.0000, 0.0040, 0.8541),
        (1.0000, 0.0040, 0.6774),
        (1.0000, 0.0040, 0.5890),
        (1.0000, 0.0040, 0.4124),
        (1.0000, 0.0040, 0.3240),
        (1.0000, 0.0040, 0.1473),
        (0.9190, 0.1564, 0.2476),
        (0.7529, 0.3782, 0.4051),
        (0.6699, 0.4477, 0.4584),
        (0.5200, 0.5200, 0.5200),
        (0.4595, 0.4595, 0.4595),
        (0.4100, 0.4100, 0.4100),
        (0.3706, 0.3706, 0.3706),
        (0.2000, 0.2000, 0.2000),
        (     0, 0, 0),
    ]

    l_colors_dwt = [
        (1, 0.134442687034607, 0.00207612453959882),
        (1, 0.531705200672150, 0.00242214533500373),
        (1, 0.928692817687988, 0.00276816613040864),
        (0.678515970706940, 1, 0.00311418692581356),
        (0.282077997922897, 1, 0.00346020772121847),
        (0.00380622851662338, 1, 0.121697589755058),
        (0.00392733560875058, 1, 0.490114688873291),
        (0.00393598619848490, 1, 0.844399273395538),
        (0.00394463678821921, 0.805243849754334, 1),
        (0.00395328737795353, 0.450971543788910, 1),
        (0.00396193796768785, 0.0967053845524788, 1),
        (0.265495806932449, 0.00397058809176087, 1),
        (0.619767010211945, 0.00397923868149519, 1),
        (0.884806394577026, 0.00398351671174169, 1),
        (1, 0.00398779427632690, 0.854077994823456),
        (1, 0.00399207184091210, 0.589043080806732)
    ]


    # interpolate colors to num cluster
    np_colors_base = np.array(l_colors_dwt)
    x = np.arange(np_colors_base.shape[0])
    itp = interp1d(x, np_colors_base, axis=0, kind='linear')

    xi = np.arange(num_clusters)
    np_colors_int = itp(xi)
    return np_colors_int

def GenOneYearDaily(yy=1981):
    'returns one generic year in a list of datetimes. Daily resolution'

    dp1 = datetime(yy,1,1)
    dp2 = datetime(yy,12,31)
    return [dp1 + timedelta(days=i) for i in range((dp2-dp1).days+1)]

def Plot_PValues(p_values, term_names, p_export=None):
    'Plot ARL/BMUS p-values'

    n_wts = p_values.shape[0]
    n_terms = p_values.shape[1]

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(16,9))

    c = ax.pcolor(p_values, cmap='coolwarm_r', clim=(0,1),
                  norm=MidpointNormalize(midpoint=0.1, vmin=0, vmax=1))
    #c.cmap.set_over('w')
    fig.colorbar(c, ax=ax)

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
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_Params(params, term_names, p_export=None):
    'Plot ARL/BMUS params'

    n_wts = params.shape[0]
    n_terms = params.shape[1]

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(16,12))

    # text table and color
    ax.matshow(params, cmap=plt.cm.bwr, origin='lower')
    for i in xrange(params.shape[1]):
        for j in xrange(params.shape[0]):
            c = params[j,i]
            ax.text(i, j, '{0:.1f}'.format(c),
                    va='center', ha='center', size=6, fontweight='bold')

    # axis
    ax.set_title('params', fontweight='bold')
    ax.set_ylabel('WT')

    ax.xaxis.tick_bottom()
    #ax.set_xticks(np.arange(n_terms), minor=True)
    ax.set_xticks(np.arange(n_terms), minor=False)
    ax.set_xticklabels(term_names, minor=False, rotation=90)

    #ax.set_yticks(np.arange(n_wts), minor=True)
    ax.set_yticks(np.arange(n_wts), minor=False)
    ax.set_yticklabels(np.arange(n_wts)+1, minor=False)

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Generate_PerpYear_Matrix(num_clusters, bmus_values, bmus_dates, num_sim=1):
    'Calculates and returns matrix for stacked bar plotting'

    # TODO BMUS_DATES HAS TO BE DATETIME 
    # TODO: RALENTIZA MUCHO, CAMBIAR ESTO PARA QUE FUNCIONE CON DATETIME Y CON
    # NUMPY.DATETIME64 RAPIDO. afecta a sacar el mes y el dia

    # TODO: doc: bmus_values has to be 2D (time, nsim)

    # generate perpetual year list
    list_pyear = GenOneYearDaily()

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
            _, bb = np.where([(j+1 == b)])

            m_plot[j,i] = float(len(bb)/float(num_sim))/len(s)

    return m_plot

def Generate_Covariate_Matrix(
    bmus_values, covar_values,
    bmus_dates, covar_dates,
    num_clusters, covar_rng, num_sim=1):
    'Calculates and returns matrix for stacked bar plotting'

    # generate aux arrays
    bmus_years = [d.year for d in bmus_dates]
    m_plot = np.zeros((num_clusters, len(covar_rng)-1)) * np.nan

    for i in range(len(covar_rng)-1):

        # find years inside range
        _, s = np.where(
            [(covar_values >= covar_rng[i]) & (covar_values <= covar_rng[i+1])]
        )
        ys = [covar_dates[x].year for x in s]

        # find data inside years found
        sb = np.where(np.in1d(bmus_years, ys))[0]

        b = bmus_values[sb,:]
        b = b.flatten()

        for j in range(num_clusters):
            _, bb = np.where([(j+1 == b)])
            if len(sb) > 0:
                m_plot[j,i] = float(len(bb)/float(num_sim))/len(sb)
            else:
                m_plot[j,i] = 0

    return m_plot

def Plot_PerpYear(bmus_values, bmus_dates, num_clusters, num_sim=1,
                  p_export=None):
    'Plots ARL bmus simulated in a perpetual_year stacked bar chart'


    # get cluster colors for stacked bar plot
    np_colors_int = GetClusterColors(num_clusters)

    # generate plot matrix
    m_plot = Generate_PerpYear_Matrix(
        num_clusters, bmus_values, bmus_dates, num_sim)

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(16,9))

    bottom_val = np.zeros(m_plot[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot[r,:]
        plt.bar(
            range(365), row_val, bottom=bottom_val,
            width=1, color = np_colors_int[r]
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
        fig.savefig(p_export, dpi=128)
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
    delta = 5
    n_rng = 7

    covar_rng = np.linspace(
        np.min(covar_values)-delta,
        np.max(covar_values)+delta,
        n_rng
    )
    interval = covar_rng[1]-covar_rng[0]

    # generate plot matrix
    m_plot = Generate_Covariate_Matrix(
        bmus_values, covar_values,
        bmus_dates, covar_dates,
        num_clusters, covar_rng, num_sims)

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(16,12))
    x_val = covar_rng[:-1]

    bottom_val = np.zeros(m_plot[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot[r,:]
        plt.bar(
            x_val, row_val, bottom=bottom_val,
            width=interval, color = np_colors_int[r]
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
        fig.savefig(p_export, dpi=128)
        plt.close()


def Plot_Compare_PerpYear(num_clusters,
                          bmus_values_sim, bmus_dates_sim,
                          bmus_values_hist, bmus_dates_hist,
                          n_sim = 1, p_export=None):
    'Plot simulated - historical bmus comparison in a perpetual year'

    # get cluster colors for stacked bar plot
    np_colors_int = GetClusterColors(num_clusters)

    # generate plot matrix
    m_plot_hist = Generate_PerpYear_Matrix(
        num_clusters, bmus_values_hist, bmus_dates_hist, num_sim=1)
    m_plot_sim = Generate_PerpYear_Matrix(
        num_clusters, bmus_values_sim, bmus_dates_sim, num_sim=n_sim)

    # plot figure
    fig, (ax_hist, ax_sim) = plt.subplots(2,1, figsize=(16,9))
    #x_val = GenOneYearDaily()
    x_val = range(365)

    # plot sim
    bottom_val = np.zeros(m_plot_sim[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot_sim[r,:]
        ax_sim.bar(
            x_val, row_val, bottom=bottom_val,
            width=1, color = np_colors_int[r]
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
    ax_sim.set_xlim(1, 365)
    ax_sim.set_ylim(0, 1)
    ax_sim.set_title('Simulation')
    ax_sim.set_ylabel('')

    ax_hist.set_xlim(1, 365)
    ax_hist.set_ylim(0, 1)
    ax_hist.set_title('Historical')
    ax_hist.set_ylabel('')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

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
    delta = 5
    n_rng = 7

    covar_rng = np.linspace(
        np.min([np.min(cov_values_hist), np.min(cov_values_sim)])-delta,
        np.max([np.max(cov_values_hist), np.max(cov_values_sim)])+delta,
        n_rng
    )
    interval = covar_rng[1]-covar_rng[0]

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
    fig, (ax_hist, ax_sim) = plt.subplots(2,1, figsize=(16,12))
    x_val = covar_rng[:-1]

    # sim
    bottom_val = np.zeros(m_plot_sim[1,:].shape)
    for r in range(num_clusters):
        row_val = m_plot_sim[r,:]
        ax_sim.bar(
            x_val, row_val, bottom=bottom_val,
            width=interval, color = np_colors_int[r]
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
    ax_sim.set_xlabel(name_covar)

    ax_hist.set_xlim(np.min(x_val)-interval/2, np.max(x_val)+interval/2)
    ax_hist.set_ylim(0, 1)
    ax_hist.set_title('Historical')
    ax_hist.set_ylabel('')
    ax_sim.set_xlabel(name_covar)

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=128)
        plt.close()
