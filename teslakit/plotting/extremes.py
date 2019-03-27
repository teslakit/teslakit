#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from lib.custom_dateutils import date2datenum as d2d

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

