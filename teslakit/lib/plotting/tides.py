#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

# fig aspect and size
_faspect = (1+5**0.5)/2.0
_fsize = 7

def Plot_AstronomicalTide(time, atide, p_export=None):
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

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_ValidateTTIDE(time, atide, atide_ttide, p_export=None):
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

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_Tide_SLR(time, atide, slr, p_export=None):
    'Plots gauge tide temporal series and SLR'

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))
    plt.plot(
        time, atide, '-k',
        linewidth = 0.3, label = 'tide'
    )
    plt.plot(
        time, slr, '-r',
        linewidth = 1, label = 'SLR'
    )
    axs.legend()
    plt.xlim(time[0], time[-1])
    plt.title('Tide with Sea Level Rise')
    plt.xlabel('time')
    plt.ylabel('Sea Level (m)')

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_Tide_RUNM(time, atide, slr, p_export=None):
    'Plots gauge tide temporal series and runm'

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))
    plt.plot(
        time, atide, '-k',
        linewidth = 0.3, label = 'tide'
    )
    plt.plot(
        time, slr, '-r',
        linewidth = 1, label = 'runm'
    )
    axs.legend()
    plt.xlim(time[0], time[-1])
    plt.title('Tide with Running Mean ')
    plt.xlabel('time')
    plt.ylabel('Sea Level (m)')

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_Tide_MMSL(
    tide_time, tide_tide,
    mmsl_time, mmsl_tide, p_export=None):
    'Plots gauge tide temporal series and mmsl'

    # plot figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))
    plt.plot(
        tide_time, tide_tide, '-k',
        linewidth = 0.3, label = 'tide'
    )
    plt.plot(
        mmsl_time, mmsl_tide, '-r',
        linewidth = 1, label = 'mmsl'
    )
    axs.legend()
    plt.xlim(mmsl_time[0], mmsl_time[-1])
    plt.title('Tide - MMSL')
    plt.xlabel('time')
    plt.ylabel('Sea Level (m)')

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

