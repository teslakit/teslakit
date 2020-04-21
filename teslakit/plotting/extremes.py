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
from scipy.stats import  gumbel_l, genextreme

# import constants
from .config import _faspect, _fsize, _fdpi

# TODO: REFACTOR CHROM Y SIGMA

def Plot_GEVParams(xda_gev_var, show=True):
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

    for c, par in enumerate(params):
        ax = axs[c]

        par_values = xda_gev_var.sel(parameter=par).values[:]
        rr_pv = np.flipud(np.reshape(par_values,(ss,ss)).T)

        if par in d_minmax.keys():
            cl = d_minmax[par]
        else:
            cl = (np.min(par_values), np.max(par_values))

        cc=ax.pcolor(rr_pv, cmap='coolwarm_r', clim=cl, edgecolor='k')
        fig.colorbar(cc, ax=ax)

        # add grid and title
        ax.set_title(par)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    ttl = 'GEV - {0}'.format(name)
    fig.suptitle(ttl, fontweight='bold', fontsize=14)

    # show
    if show: plt.show()
    return fig

def Plot_ChromosomesProbs(xds_chrom, show=True):
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

        ttl = 'Chromosomes Probabilities - WTs'
        fig.suptitle(ttl, fontweight='bold', fontsize=14)

    # show
    if show: plt.show()
    return fig

def Plot_SigmaCorrelation(xds_chrom, d_sigma, show=True):
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
            al = np.min([1.0, np.absolute(sigma_plot[c_wt, c_c]) / maxcorr])
            ax.add_patch(
                plt.Circle(
                    cntr,
                    radius = maxsize * np.absolute(sigma_plot[c_wt, c_c]),
                    edgecolor = e_cl,
                    facecolor = mk_cl,
                    alpha = al
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

        ttl = 'Sigma Correlation - WTs'
        fig.suptitle(ttl, fontweight='bold', fontsize=14)

    # show
    if show: plt.show()
    return fig

def Plot_Schemaball(xds_data):
    # TODO
    return None


def axplot_RP(ax, t_h, v_h, tg_h, vg_h, t_s, v_s, var_name, sim_percentile=95):
    'axes plot return period historical vs simulation'

    # historical maxima
    ax.semilogx(
        t_h, v_h, 'ok',
        markersize = 3, label = 'Historical',
        zorder=9,
    )

    # TODO: fit historical to gev
    # historical GEV fit
    #ax.semilogx(
    #    tg_h, vg_h, '-b',
    #    label = 'Historical - GEV Fit',
    #)

    # simulation maxima - mean
    mn = np.mean(v_s, axis=0)
    ax.semilogx(
        t_s, mn, '-r',
        linewidth = 2, label = 'Simulation (mean)',
        zorder=8,
    )

    # simulation maxima percentile 95% and 05%
    p95 = np.percentile(v_s, sim_percentile, axis=0,)
    p05 = np.percentile(v_s, 100-sim_percentile, axis=0,)

    ax.semilogx(
        t_s, p95, linestyle='-', color='grey',
        linewidth = 2, #label = 'Simulation (95% percentile)',
    )

    ax.semilogx(
        t_s, p05, linestyle='-', color='grey',
        linewidth = 2, # label = 'Simulation (05% percentile)',
    )
    ax.fill_between(
        t_s, p05, p95, color='lightgray',
        label = 'Simulation (05 - 95 percentile)'
    )

    # customize axs
    ax.legend(loc='lower right')
    ax.set_title('Annual Maxima', fontweight='bold')
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel('{0}'.format(var_name))
    ax.set_xlim(left=10**0, right=np.max(np.concatenate([t_h,t_s])))
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.grid(which='both')

def Plot_ReturnPeriodValidation(xds_hist, xds_sim, sim_percentile=95, show=True):
    'Plot Return Period historical - simulation validation'

    # aux func for calculating rp time
    def t_rp(time_y):
        ny = len(time_y)
        return np.array([1/(1-(n/(ny+1))) for n in np.arange(1,ny+1)])

    # aux func for gev fit
    # TODO: fix it
    def gev_fit(var_fit):
        c = -0.1
        vv = np.linspace(0,10,200)

        sha_g, loc_g, sca_g =  genextreme.fit(var_fit, c)
        pg = genextreme.cdf(vv, sha_g, loc_g, sca_g)

        ix = pg > 0.1
        vv = vv[ix]
        ts = 1/(1 - pg[ix])

        # TODO gev params 95% confidence intervals

        return ts, vv

    # clean nans
    t_r = xds_hist.year.values[:]
    v_r = xds_hist.values[:]

    ix_nan = np.isnan(v_r)
    t_r = t_r[~ix_nan]
    v_r = v_r[~ix_nan]

    # RP calculation, var sorting historical
    t_h = t_rp(t_r)
    v_h = np.sort(v_r)

    # GEV fit historical
    #tg_h, vg_h = gev_fit(v_h)
    tg_h, vg_h = [],[]

    # RP calculation, var sorting simulation
    t_s = t_rp(xds_sim.year.values[:-1])  # remove last year*
    v_s = np.sort(xds_sim.values[:,:-1])  # remove last year*

    # figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    axplot_RP(
        axs,
        t_h, v_h, tg_h, vg_h,
        t_s, v_s,
        xds_sim.name,
        sim_percentile=sim_percentile,
    )

    # show and return figure
    if show: plt.show()
    return fig


# TODO: revisar funciones _simulations

def axplot_RP_Sims(ax, tups_time_val, var_name):
    'axes plot return period historical vs simulation'


    lbls_cls = [
        #('Simulation', 'black'),
        ('Simulation - El Niño', 'red'),
        ('Simulation - La Niña' , 'blue'),
    ]

    for (t_s, v_s), (lb, cl) in zip(tups_time_val, lbls_cls):


        # simulation maxima - mean
        mn = np.mean(v_s, axis=0)
        ax.semilogx(
            t_s, mn, linestyle= '-', color=cl,
            linewidth = 2, label = lb,
            zorder=8,
        )

        # simulation maxima percentile 95% and 05%
        p95 = np.percentile(v_s, 95, axis=0,)
        p05 = np.percentile(v_s, 5, axis=0,)

        ax.semilogx(
            t_s, p95, linestyle='--', color=cl, alpha=0.05,
            linewidth = 2, #label = 'Simulation (95% percentile)',
        )

        ax.semilogx(
            t_s, p05, linestyle='--', color=cl, alpha=0.05,
            linewidth = 2, # label = 'Simulation (05% percentile)',
        )
        ax.fill_between(
            t_s, p05, p95, color=cl,alpha=0.05,
            #label = 'Simulation (05 - 95 percentile)'
        )

    # customize axs
    ax.legend(loc='lower right')
    ax.set_title('Annual Maxima', fontweight='bold')
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel('{0}'.format(var_name))
    ax.set_xlim(left=10**0, right=np.max(t_s))
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.grid(which='both')


def Plot_ReturnPeriod_Simulations(l_xds_sim, show=True):
    'Plot Return Period historical - simulation validation'

    # aux func for calculating rp time
    def t_rp(time_y):
        ny = len(time_y)
        return np.array([1/(1-(n/(ny+1))) for n in np.arange(1,ny+1)])

    # RP calculation, var sorting simulation
    tups_time_val = []
    for xds_sim in l_xds_sim:
        t_s = t_rp(xds_sim.year.values[:-1])  # remove last year*
        v_s = np.sort(xds_sim.values[:,:-1])  # remove last year*

        tups_time_val.append((t_s, v_s))

    # figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    axplot_RP_Sims(
        axs,
        tups_time_val,
        xds_sim.name,
    )

    # show and return figure
    if show: plt.show()
    return fig



# TODO: revisar funciones _v2

def axplot_RP_v2(ax, t_h, v_h, tg_h, vg_h, t_s, v_s, t_s2, v_s2, var_name):
    'axes plot return period historical vs simulation'

    # historical maxima
    ax.semilogx(
        t_h, v_h, 'ok',
        markersize = 3, label = 'Historical',
        zorder=9,
    )

    # TODO: fit historical to gev
    # historical GEV fit
    #ax.semilogx(
    #    tg_h, vg_h, '-b',
    #    label = 'Historical - GEV Fit',
    #)

    # simulation maxima - mean
    mn = np.mean(v_s, axis=0)
    ax.semilogx(
        t_s, mn, '-r',
        linewidth = 2, label = 'Simulation (mean)',
        zorder=8,
    )
    # simulation maxima - mean
    mn2 = np.mean(v_s2, axis=0)
    ax.semilogx(
        t_s2, mn2, '-b',
        linewidth = 2, label = 'Simulation Climate Change (mean)',
        zorder=8,
    )

    # simulation maxima percentile 95% and 05%
    p95 = np.percentile(v_s, 95, axis=0,)
    p05 = np.percentile(v_s, 5, axis=0,)

    ax.semilogx(
        t_s, p95, linestyle='-', color='grey',
        linewidth = 2, #label = 'Simulation (95% percentile)',
    )

    ax.semilogx(
        t_s, p05, linestyle='-', color='grey',
        linewidth = 2, # label = 'Simulation (05% percentile)',
    )
    ax.fill_between(
        t_s, p05, p95, color='lightgray',
        label = 'Simulation (05 - 95 percentile)'
    )

    # simulation maxima percentile 95% and 05%
    p95 = np.percentile(v_s2, 95, axis=0,)
    p05 = np.percentile(v_s2, 5, axis=0,)

    ax.semilogx(
        t_s2, p95, linestyle='-', color='skyblue',
        linewidth = 2, #label = 'Simulation (95% percentile)',
    )

    ax.semilogx(
        t_s2, p05, linestyle='-', color='skyblue',
        linewidth = 2, # label = 'Simulation (05% percentile)',
    )
    ax.fill_between(
        t_s2, p05, p95, color='skyblue', alpha=0.5,
        label = 'Simulation Climate Change (05 - 95 percentile)'
    )

    # customize axs
    ax.legend(loc='lower right')
    ax.set_title('Annual Maxima', fontweight='bold')
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel('{0}'.format(var_name))
    ax.set_xlim(left=10**0, right=np.max(np.concatenate([t_h,t_s])))
    ax.tick_params(axis='both', which='both', top=True, right=True)


def Plot_ReturnPeriodValidation_v2(xds_hist, xds_sim, xds_sim2, show=True):
    'Plot Return Period historical - simulation validation'

    # aux func for calculating rp time
    def t_rp(time_y):
        ny = len(time_y)
        return np.array([1/(1-(n/(ny+1))) for n in np.arange(1,ny+1)])

    # aux func for gev fit
    # TODO: fix it
    def gev_fit(var_fit):
        c = -0.1
        vv = np.linspace(0,10,200)

        sha_g, loc_g, sca_g =  genextreme.fit(var_fit, c)
        pg = genextreme.cdf(vv, sha_g, loc_g, sca_g)

        ix = pg > 0.1
        vv = vv[ix]
        ts = 1/(1 - pg[ix])

        # TODO gev params 95% confidence intervals

        return ts, vv

    # RP calculation, var sorting historical
    t_h = t_rp(xds_hist.year.values[:])
    v_h = np.sort(xds_hist.values[:])

    # GEV fit historical
    tg_h, vg_h = gev_fit(v_h)

    # RP calculation, var sorting simulation
    t_s = t_rp(xds_sim.year.values[:-1])  # remove last year
    v_s = np.sort(xds_sim.values[:,:-1])  # remove last year

    t_s2 = t_rp(xds_sim2.year.values[:-1])  # remove last year
    v_s2 = np.sort(xds_sim2.values[:,:-1])  # remove last year

    # figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    axplot_RP_v2(
        axs,
        t_h, v_h, tg_h, vg_h,
        t_s, v_s,
        t_s2, v_s2,
        xds_sim.name,
    )

    # show and return figure
    if show: plt.show()
    return fig

