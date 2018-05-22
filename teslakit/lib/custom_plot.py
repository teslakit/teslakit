#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import interp1d
import calendar
from datetime import datetime, timedelta

# TODO: CREAR FUNCION CustomColormap con datos y el interp

def Plot_EOFs_Annual(xds_PCA, lon, y1, y2, m1, m2, n_plot, p_export=None):
    '''
    Plot annual EOFs for 3D predictors

    xds_PCA:
        (n_components, n_components) PCs
        (n_components, n_features) EOFs
        (n_components, ) variance
    lon: predictor longitude values
    y1,y2,m1,m2: PCA time parameters
    n_plot: number of EOFs plotted

    show plot or saves figure to p_export
    '''

    # PCA data
    variance = xds_PCA['variance'].values
    EOFs = np.transpose(xds_PCA['EOFs'].values)
    PCs = np.transpose(xds_PCA['PCs'].values)

    # mesh data
    len_x = len(lon)

    # time data
    years = range(y1, y2+1)
    l_months = [calendar.month_name[x] for x in range(1,13)]
    ylbl = l_months[m1-1:] + l_months[:m2]


    # percentage of variance each field explains
    n_percent = variance / np.sum(variance)

    for it in range(n_plot):

        # plot figure
        fig = plt.figure(figsize=(16,9))

        # map of the spatial field
        spatial_fields = EOFs[:,it]*np.sqrt(variance[it])

        # reshape from vector to matrix with separated months
        C = np.reshape(spatial_fields[:len_x*12], (12, len_x)).transpose()

        # eof cmap
        ax1 = plt.subplot2grid((6, 6), (0, 0), colspan=6, rowspan=4)
        plt.pcolormesh(np.transpose(C), cmap='RdBu', shading='gouraud')
        plt.clim(-1,1)
        plt.title('EOF #{0}  ---  {1:.2f}%'.format(it+1,n_percent[it]*100))
        ax1.set_xticklabels([str(x) for x in lon])
        ax1.set_yticklabels(ylbl)

        # time series
        ax2 = plt.subplot2grid((6, 6), (5, 0), colspan=6, rowspan=2)
        plt.plot(years, PCs[it,:]/np.sqrt(variance[it]))
        plt.xlim(years[0], years[-1])

        # show / export
        if not p_export:
            plt.show()

        else:
            if not op.isdir(p_export):
                os.makedirs(p_export)
            p_expi = op.join(p_export, 'EOFs_{0}'.format(it+1))
            fig.savefig(p_expi, dpi=96)
            plt.close()


def Plot_MJOphases(rmm1, rmm2, phase, p_export=None):
    'Plot MJO data separated by phase'

    # parameters for custom plot
    size_points = 0.2
    size_lines = 0.8
    l_colors_phase = [
        (1, 0, 0),
        (0.6602, 0.6602, 0.6602),
        (1.0, 0.4961, 0.3125),
        (0, 1, 0),
        (0.2539, 0.4102, 0.8789),
        (0, 1, 1),
        (1, 0.8398, 0),
        (0.2930, 0, 0.5078)]
    color_lines_1 = (0.4102, 0.4102, 0.4102)


    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(9,9))
    ax.scatter(rmm1, rmm2, c='b', s=size_points)

    # plot data by phases
    for i in range(1,9):
        ax.scatter(
            rmm1.where(phase==i),
            rmm2.where(phase==i),
            c=l_colors_phase[i-1],
            s=size_points)

    # plot sectors
    ax.plot([-4,4],[-4,4], color='k', linewidth=size_lines)
    ax.plot([-4,4],[4,-4], color='k', linewidth=size_lines)
    ax.plot([-4,4],[0,0],  color='k', linewidth=size_lines)
    ax.plot([0,0], [-4,4], color='k', linewidth=size_lines)

    # axis
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('RMM1')
    plt.ylabel('RMM2')
    ax.set_aspect('equal')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()

def Plot_MJOCategories(rmm1, rmm2, categ, p_export=None):
    'Plot MJO data separated by 25 categories'

    # parameters for custom plot
    size_lines = 0.8
    color_lines_1 = (0.4102, 0.4102, 0.4102)
    # TODO: COLORES PARA 25 CATEGORIAS, NO PARA N
    l_colors_categ = [
        (0.527343750000000, 0.804687500000000, 0.979166666666667),
        (0, 0.746093750000000, 1),
        (0.253906250000000, 0.410156250000000, 0.878906250000000),
        (0, 0, 0.800781250000000),
        (0, 0, 0.542968750000000),
        (0.273437500000000, 0.507812500000000, 0.703125000000000),
        (0, 0.804687500000000, 0.816406250000000),
        (0.250000000000000, 0.875000000000000, 0.812500000000000),
        (0.500000000000000, 0, 0),
        (0.542968750000000, 0.269531250000000, 0.0742187500000000),
        (0.820312500000000, 0.410156250000000, 0.117187500000000),
        (1, 0.839843750000000, 0),
        (1, 0.644531250000000, 0),
        (1, 0.269531250000000, 0),
        (1, 0, 0),
        (0.695312500000000, 0.132812500000000, 0.132812500000000),
        (0.500000000000000, 0, 0.500000000000000),
        (0.597656250000000, 0.195312500000000, 0.796875000000000),
        (0.726562500000000, 0.332031250000000, 0.824218750000000),
        (1, 0, 1),
        (0.480468750000000, 0.406250000000000, 0.929687500000000),
        (0.539062500000000, 0.167968750000000, 0.882812500000000),
        (0.281250000000000, 0.238281250000000, 0.542968750000000),
        (0.292968750000000, 0, 0.507812500000000),
        (0.660156250000000, 0.660156250000000, 0.660156250000000),
    ]

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(9,9))

    # plot sectors
    ax.plot([-4,4],[-4,4], color='k', linewidth=size_lines, zorder=9)
    ax.plot([-4,4],[4,-4], color='k', linewidth=size_lines, zorder=9)
    ax.plot([-4,4],[0,0],  color='k', linewidth=size_lines, zorder=9)
    ax.plot([0,0], [-4,4], color='k', linewidth=size_lines, zorder=9)

    # plot circles
    R = [1, 1.5, 2.5]

    for rr in R:
        ax.add_patch(
            patches.Circle(
                (0,0),
                rr,
                color='k',
                linewidth=size_lines,
                fill=False,
                zorder=9)
        )
    ax.add_patch(
        patches.Circle((0,0),R[0],fc='w',fill=True, zorder=10))

    # plot data by categories
    for i in range(1,25):
        if i>8: size_points = 0.2
        else: size_points = 1.7

        ax.scatter(
            rmm1.where(categ==i),
            rmm2.where(categ==i),
            c=l_colors_categ[i-1],
            s=size_points)
    ax.scatter(
        rmm1.where(categ==25),
        rmm2.where(categ==25),
        c=l_colors_categ[i-1],
        s=0.2,
    zorder=11)

    # axis
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('RMM1')
    plt.ylabel('RMM2')
    ax.set_aspect('equal')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()


def Plot_ARL_PValues(p_values, term_names, p_export=None):
    'Plot ARL/BMUS p-values'

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(16,9))

    # TODO: DESACHER
    #c = ax.pcolor(p_values, cmap='inferno', vmin=0, vmax=0.1)
    c = ax.pcolor(p_values, cmap='inferno', vmin=0, vmax=1)
    c.cmap.set_over('w')
    fig.colorbar(c, ax=ax)

    # axis
    ax.set_title('p-value', fontweight='bold')
    ax.set_ylabel('WT')
    plt.xticks(np.arange(len(term_names))+0.5, term_names, rotation=90)
    [t.label.set_fontsize(8) for t in ax.xaxis.get_major_ticks()]

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()

def Plot_ARL_Params(params, term_names, p_export=None):
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

def Plot_ARL_PerpYear(bmus_values, bmus_dates, num_clusters, num_sims):
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

    # TODO: GUARDAR EL INTERPOLADOR EN UNA FUNCION
    # interpolate colors to num cluster
    np_colors_base = np.array(l_colors_dwt)
    x = np.arange(np_colors_base.shape[0])
    itp = interp1d(x, np_colors_base, axis=0, kind='linear')

    xi = np.arange(num_clusters)
    np_colors_int =  itp(xi)

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
    plt.figure(1)
    ax = plt.subplot(111)

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

    # show
    plt.show()


def Plot_CSIRO_Stations(xds_stations, p_export=None):
    'Plot CSIRO spec station location over the world'
    # TODO: mejorar, introducir variables opcionales 

    # xds_station lon lat
    lon = xds_stations.longitude.values[:]
    lat = xds_stations.latitude.values[:]
    nms = xds_stations.station_name[:]

    # basemap
    m = Basemap(
        #width=12000000,
        #height=9000000,
        #resolution=None,
        projection='merc',
        llcrnrlon=311, urcrnrlon=312,
        llcrnrlat=-28, urcrnrlat=-27,
    )

    # draw parallels.
    parallels = np.arange(-90.,90.,10.)
    m.drawparallels(parallels, labels=[1,0,0,0],fontsize=6)
    # draw meridians
    meridians = np.arange(0.,360.,10.)
    m.drawmeridians(meridians, labels=[0,0,0,1],fontsize=6)

    # add stations
    m.scatter(lon, lat, s=6, c='r', latlon=True)

    # add shader
    m.shadedrelief()

    plt.show()
    return

    # TODO show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=128)
        plt.close()


