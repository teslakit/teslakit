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

from lib.custom_dateutils import xds2datetime


def Plot_EOFs_latavg(xds_PCA, n_plot, p_export=None):
    '''
    Plot annual EOFs for 3D predictors

    xds_PCA:
        (n_components, n_components) PCs
        (n_components, n_features) EOFs
        (n_components, ) variance

        (n_lon, ) pred_lon: predictor longitude values

        attrs: y1, y2, m1, m2: PCA time parameters
        method: latitude averaged

    n_plot: number of EOFs plotted

    show plot or saves figure to p_export
    '''

    # PCA data
    variance = xds_PCA['variance'].values
    EOFs = np.transpose(xds_PCA['EOFs'].values)
    PCs = np.transpose(xds_PCA['PCs'].values)

    # PCA latavg metadata
    y1 = xds_PCA.attrs['y1']
    y2 = xds_PCA.attrs['y2']
    m1 = xds_PCA.attrs['m1']
    m2 = xds_PCA.attrs['m2']
    lon = xds_PCA['pred_lon'].values

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

def Plot_EOFs_EstelaPred(xds_PCA, n_plot, p_export=None):
    '''
    Plot annual EOFs for 3D predictors

    xds_PCA:
        (n_components, n_components) PCs
        (n_components, n_features) EOFs
        (n_components, ) variance

        (n_lon, ) pred_lon: predictor longitude values
        (n_lat, ) pred_lat: predictor latitude values
        (n_time, ) pred_time: predictor time values

        method: gradient + estela

    n_plot: number of EOFs plotted

    show plot or saves figure to p_export
    '''

    # PCA data
    variance = xds_PCA['variance'].values
    EOFs = np.transpose(xds_PCA['EOFs'].values)
    PCs = np.transpose(xds_PCA['PCs'].values)
    data_pos = xds_PCA['pred_data_pos']  # for handling nans
    pca_time = xds_PCA['pred_time']
    pred_name = xds_PCA.attrs['pred_name']

    # PCA lat lon metadata
    lon = xds_PCA['pred_lon'].values
    lat = xds_PCA['pred_lat'].values

    # percentage of variance each field explains
    n_percent = variance / np.sum(variance)

    for it in range(n_plot):

        # plot figure
        fig = plt.figure(figsize=(16,9))

        # get vargrd 
        var_grd_1d = EOFs[:,it]*np.sqrt(variance[it])

        # insert nans in data
        base = np.nan * np.ones(data_pos.shape)
        base[data_pos] = var_grd_1d
        var = base[:len(base)/2]
        grd = base[len(base)/2:]

        # reshape data to grid
        C1 = np.reshape(var, (len(lon), len(lat)))
        C2 = np.reshape(grd, (len(lon), len(lat)))

        # eof cmap
        ax1 = plt.subplot2grid((6, 6), (0, 0), colspan=3, rowspan=4)
        plt.pcolormesh(
            np.flipud(np.transpose(C1)), cmap='RdBu', shading='gouraud')
        plt.clim(-1,1)
        #fig.colorbar(pm, ax=ax1)
        plt.suptitle('EOF #{0}  ---  {1:.2f}%'.format(it+1,n_percent[it]*100))
        plt.title(pred_name)
        ax1.set_xticklabels([str(x) for x in lon])
        ax1.set_yticklabels([str(x) for x in lat])

        ax2 = plt.subplot2grid((6, 6), (0, 3), colspan=3, rowspan=4)
        plt.pcolormesh(
            np.flipud(np.transpose(C2)), cmap='RdBu', shading='gouraud')
        plt.title('{0} gradient'.format(pred_name))
        plt.clim(-1,1)
        #fig.colorbar(pm, ax=ax2)
        ax2.set_xticklabels([str(x) for x in lon])
        ax2.set_yticklabels(['' for x in lat])
        #ax2.get_yaxis().set_visible(False)

        # time series
        ax3 = plt.subplot2grid((6, 6), (5, 0), colspan=6, rowspan=2)
        plt.plot(pca_time, PCs[it,:]/np.sqrt(variance[it]))
        dl_1 = xds2datetime(pca_time[0])
        dl_2 = xds2datetime(pca_time[-1])
        plt.xlim(dl_1, dl_2)


        # show / export
        if not p_export:
            plt.show()

        else:
            if not op.isdir(p_export):
                os.makedirs(p_export)
            p_expi = op.join(p_export, 'EOFs_{0}'.format(it+1))
            fig.savefig(p_expi, dpi=96)
            plt.close()
