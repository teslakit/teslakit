#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def Plot_PredictorEOFs(xds_PCA, n_plot):
    '''
    Plot EOFs
    '''
    # TODO: DOC

    # PCA data
    variance = xds_PCA['variance'].values
    EOFs = np.transpose(xds_PCA['EOFs'].values)
    PCs = np.transpose(xds_PCA['PCs'].values)

    years = xds_PCA['_years'].values
    lon = xds_PCA['_longitude'].values
    len_x = len(lon)

    # percentage of variance each field explains
    n_percent = variance / np.sum(variance)

    for it in range(n_plot):

        # map of the spatial field
        spatial_fields = EOFs[:,it]*np.sqrt(variance[it])

        # reshape from vector to matrix with separated months
        C = np.reshape(spatial_fields[:len_x*12], (12, len_x)).transpose()

        # TODO: meshgrid para usar meses / longitude en axis

        # eof cmap
        ax1 = plt.subplot2grid((6, 6), (0, 0), colspan=6, rowspan=4)
        plt.pcolormesh(np.transpose(C), cmap='RdBu', shading='gouraud')
        plt.clim(-1,1)
        plt.title('EOF #{0}  ---  {1:.2f}%'.format(it+1,n_percent[it]*100))

        # time series
        ax2 = plt.subplot2grid((6, 6), (5, 0), colspan=6, rowspan=2)
        plt.plot(years, PCs[it,:]/np.sqrt(variance[it]))
        plt.xlim(years[0], years[-1])

        # SHOW
        plt.show()



