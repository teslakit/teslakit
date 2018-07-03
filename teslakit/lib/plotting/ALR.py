#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime, timedelta


def Plot_PValues(p_values, term_names, p_export=None):
    'Plot ARL/BMUS p-values'

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(16,9))

    c = ax.pcolor(p_values, cmap='inferno', vmin=0, vmax=0.1)
    #c = ax.pcolor(p_values, cmap='inferno', vmin=0, vmax=1)
    c.cmap.set_over('w')
    fig.colorbar(c, ax=ax)

    # axis
    ax.set_title('p-value', fontweight='bold')
    ax.set_ylabel('WT')
    ax.xaxis.tick_bottom()
    plt.xticks(np.arange(len(term_names))+0.5, term_names, rotation=90)
    [t.label.set_fontsize(8) for t in ax.xaxis.get_major_ticks()]

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_Params(params, term_names, p_export=None):
    'Plot ARL/BMUS params'

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
    plt.xticks(np.arange(len(term_names))+0.5, term_names, rotation=90)
    [t.label.set_fontsize(8) for t in ax.xaxis.get_major_ticks()]

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

def Plot_PerpYear(bmus_values, bmus_dates, num_clusters, num_sims,
                  p_export=None):
    'Plots ARL bmus simulated in a perpetual_year stacked bar chart'

    # parameters for custom plot
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

    # interpolate colors to num cluster
    np_colors_base = np.array(l_colors_dwt)
    x = np.arange(np_colors_base.shape[0])
    itp = interp1d(x, np_colors_base, axis=0, kind='linear')

    xi = np.arange(num_clusters)
    np_colors_int =  itp(xi)

    # bmus_values has to be 1D
    bmus_values = np.squeeze(bmus_values)

    # generate perpetual year list
    dp1 = datetime(1981,1,1)
    dp2 = datetime(1981,12,31)
    list_pyear = [dp1 + timedelta(days=i) for i in range((dp2-dp1).days+1)]

    # generate aux arrays
    m_plot = np.zeros((num_clusters, len(list_pyear))) * np.nan
    bmus_dates_months = np.array([d.month for d in bmus_dates])
    bmus_dates_days = np.array([d.day for d in bmus_dates])

    # sort data
    for i, dpy in enumerate(list_pyear):
        _, s = np.where(
            [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
        )
        b = bmus_values[s]

        for j in range(num_clusters):
            _, bb = np.where([(j+1 == b)])

            m_plot[j,i] = float(len(bb))/len(s)

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(16,12))

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
