#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import calendar

# pip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from datetime import datetime, timedelta

# tk
from ..custom_dateutils import xds2datetime

# register matplotlib converters
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()

# fig aspect and size
_faspect = (1+5**0.5)/2.0
_fsize = 7


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
        fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

        # get vargrd 
        var_grd_1d = EOFs[:,it]*np.sqrt(variance[it])

        # insert nans in data
        base = np.nan * np.ones(data_pos.shape)
        base[data_pos] = var_grd_1d

        var = base[:int(len(base)/2)]
        grd = base[int(len(base)/2):]

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

def Plot_PCvsPC(xds_PC123, text=[], p_export = None):
    '''
    Plot PC1 vs PC2 vs PC3

    xds_PD123
        (dim,) PC1
        (dim,) PC2
        (dim,) PC3

        (dim,) text

    show plot or saves figure to p_export
    '''

    # get data
    pc1_val = xds_PC123.PC1.values
    pc2_val = xds_PC123.PC2.values
    pc3_val = xds_PC123.PC3.values

    # delta axis
    pc1_d = np.max([np.absolute(np.max(pc1_val)), np.absolute(np.min(pc1_val))])
    pc2_d = np.max([np.absolute(np.max(pc2_val)), np.absolute(np.min(pc2_val))])
    pc3_d = np.max([np.absolute(np.max(pc3_val)), np.absolute(np.min(pc3_val))])

    pf = 1.05
    pc1_d = pc1_d * pf
    pc2_d = pc2_d * pf
    pc3_d = pc3_d * pf

    # create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(_faspect*_fsize, _fsize))

    ax1.plot(pc2_val, pc1_val, '.r')
    ax2.plot(pc3_val, pc1_val, '.r')
    ax4.plot(pc3_val, pc2_val, '.r')
    ax3.remove()

    # text
    for p1,p2,p3,t in zip(pc1_val,pc2_val,pc3_val,text):
        ax1.text(p2,p1,t)
        ax2.text(p3,p1,t)
        ax4.text(p3,p2,t)

    # labels and customize
    fw = 'bold'
    ax1.set_xlabel('PC2', fontweight=fw)
    ax1.set_ylabel('PC1', fontweight=fw)
    ax2.set_xlabel('PC3', fontweight=fw)
    ax2.set_ylabel('PC1', fontweight=fw)
    ax4.set_xlabel('PC3', fontweight=fw)
    ax4.set_ylabel('PC2', fontweight=fw)

    ax1.set_xlim(-pc2_d, pc2_d)
    ax1.set_ylim(-pc1_d, pc1_d)
    ax2.set_xlim(-pc3_d, pc3_d)
    ax2.set_ylim(-pc1_d, pc1_d)
    ax4.set_xlim(-pc3_d, pc3_d)
    ax4.set_ylim(-pc2_d, pc2_d)

    lc = 'k'
    lls = '--'
    for ax in [ax1, ax2, ax4]:
        ax.axhline(y=0, color=lc, linestyle=lls)
        ax.axvline(x=0, color=lc, linestyle=lls)

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()

