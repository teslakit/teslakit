#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

# teslakit
from ..util.time_operations import date2datenum as d2d

# import constants
from .config import _faspect, _fsize, _fdpi

def axplot_histcompare(ax, var_fit, var_sim, color='green', n_bins=40):
    'axes plot histogram comparison between fit-sim variables'

    (_, bins, _) = ax.hist(var_fit, n_bins, weights=np.ones(len(var_fit)) / len(var_fit),
            alpha=0.9, color='white', ec='k', label = 'Historical')

    ax.hist(var_sim, bins=bins, weights=np.ones(len(var_sim)) / len(var_sim),
            alpha=0.4, color=color, ec='k', label = 'Simulation')

    # customize axes
    ax.legend(prop={'size':10})

def Plot_AstronomicalTide(time, atide, show=True):
    'Plots astronomical tide temporal series'

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))
    plt.plot(
        time, atide, '-k',
        linewidth = 0.04,
    )
    plt.xlim(time[0], time[-1])
    plt.title('Astronomical tide')
    plt.xlabel('time')
    plt.ylabel('tide (m)')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_ValidateTTIDE(time, atide, atide_ttide, show=True):
    'Compares astronomical tide and ttide prediction'

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))
    plt.plot(
        time, atide, '-k',
        linewidth = 0.04,
        label = 'data'
    )
    plt.plot(
        time, atide_ttide, '-r',
        linewidth = 0.02,
        label = 'ttide model'
    )
    plt.xlim(time[0], time[-1])
    plt.title('Astronomical tide - TTIDE validation')
    plt.xlabel('time')
    plt.ylabel('tide (m)')
    axs.legend()

    # show and return figure
    if show: plt.show()
    return fig

def axplot_sea_level(ax, sea_level_time, sea_level_val, var_time, var_val,
                     var_name):
    'axes plot sea level and other variable: running mean, slr, etc'

    ax.plot(
        sea_level_time, sea_level_val, '-k',
        linewidth = 0.2, label = 'tide'
    )
    ax.plot(
        var_time, var_val, '-r',
        linewidth = 1, label = var_name
    )

    # customize axs
    ax.legend()
    ax.set_xlim(var_time[0], var_time[-1])
    ax.set_title('Tide - {0}'.format(var_name), fontweight='bold')
    ax.set_xlabel('time')
    ax.set_ylabel('Sea Level (mm)')

def Plot_Tide_SLR(time, tide, slr, show=True):
    'Plots gauge tide temporal series and SLR'

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize/2))

    axplot_sea_level(axs, time, tide, time, slr, 'Sea Level Rise')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_Tide_RUNM(time, tide, rum, show=True):
    'Plots gauge tide temporal series and runm'

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize/2))

    axplot_sea_level(axs, time, tide, time, rum, 'Running Mean')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_Tide_MMSL(tide_time, tide_val, mmsl_time, mmsl_val, show=True):
    'Plots gauge tide temporal series and mmsl'

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize/2))

    axplot_sea_level(axs, tide_time, tide_val, mmsl_time, mmsl_val, 'MMSL')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_Validate_MMSL_tseries(
    mmsl_time, mmsl_data, mmsl_pred, show=True):
    'Plots series comparison between mmsl data and mmsl predicted with nlm'

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize/2))
    plt.plot(
        mmsl_time, mmsl_data, '-k',
        linewidth = 0.7, label = 'Historical'
    )
    plt.plot(
        mmsl_time, mmsl_pred, '-r',
        linewidth = 0.7, label = 'Prediction'
    )
    axs.axhline(linestyle='--', linewidth=0.5, color='k')
    axs.legend()
    axs.set_xlim(mmsl_time[0], mmsl_time[-1])

    ttl = 'Monthly Mean Sea Level non-linear model'
    axs.set_title(ttl, fontweight='bold')
    axs.set_xlabel('time')
    axs.set_ylabel('MMSL (mm)')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_Validate_MMSL_scatter(mmsl_data, mmsl_pred, show=True):
    'Plots scatter comparison between mmsl data and mmsl predicted with nlm'

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize/1.5, _fsize/1.5))
    axs.plot(
        mmsl_data, mmsl_pred, 'or'
    )
    axs.axhline(linestyle='--', linewidth=0.5, color='k')
    axs.axvline(linestyle='--', linewidth=0.5, color='k')

    dmin = np.min(mmsl_data)
    dmax = np.max(mmsl_data)
    axs.plot([dmin, dmax], [dmin, dmax], '--k')

    axs.set_xlabel('Historical MMSL (mm)')
    axs.set_ylabel('Simulated MMSL (mm)')
    axs.set_aspect('equal', 'box')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_MMSL_Prediction(mmsl_time, mmsl_data, show=True):
    'Plots predicted mmsl data'

    # parse time array to datetime
    mmsl_time = [d2d(x) for x in mmsl_time]

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize/2))
    axs.plot(
        mmsl_time, mmsl_data, '-g',
        linewidth = 0.5, label = 'Simulation'
    )
    axs.axhline(linestyle='--', linewidth=0.5, color='k')

    axs.legend()
    axs.set_xlim(mmsl_time[0], mmsl_time[-1])
    ttl = 'Monthly Mean Sea Level non-linear model prediction'
    axs.set_title(ttl, fontweight='bold')
    axs.set_xlabel('time')
    axs.set_ylabel('MMSL (mm)')

    # show and return figure
    if show: plt.show()
    return fig

def Plot_MMSL_Histogram(mmsl_fit, mmsl_sim, show=True):
    'Plots predicted mmsl data'

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize/2))
    axplot_histcompare(axs, mmsl_fit, mmsl_sim)

    axs.set_xlabel('MMSL (mm)')

    # show and return figure
    if show: plt.show()
    return fig

