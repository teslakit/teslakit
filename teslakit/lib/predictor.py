#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import datetime
from sklearn.decomposition import PCA

# tk libs
from lib.custom_stats import running_mean


def CalcRunningMean(xdset, pred_name, window=5):
    '''
    Calculate running average grouped by months
    Dylan methodology

    xdset: (longitude, latitude, time) pred_name
    returns xdset with new variable "pred_name_runavg"
    '''
    # TODO: MUY LENTO, OPTIMIZAR

    tempdata_runavg = np.empty(xdset[pred_name].shape)

    for lon in xdset.longitude.values:
       for lat in xdset.latitude.values:
          for mn in range(1, 13):

             # indexes
             ix_lon = np.where(xdset.longitude == lon)
             ix_lat = np.where(xdset.latitude == lat)
             ix_mnt = np.where(xdset['time.month'] == mn)

             # point running average
             time_mnt = xdset.time[ix_mnt]
             data_pnt = xdset[pred_name].loc[lon, lat, time_mnt]


             tempdata_runavg[ix_lon[0], ix_lat[0], ix_mnt[0]] = running_mean(data_pnt.values, 5)

    # store running average
    xdset['{0}_runavg'.format(pred_name)]= (
        ('longitude', 'latitude', 'time'),
        tempdata_runavg)

    return xdset

def CalcPCA_Annual_latavg(xdset, pred_name, y1, y2, m1, m2):
    '''
    Principal component analysis
    Annual data and latitude average

    xdset:
        (longitude, latitude, time), pred_name | pred_name_runavg

    returns a xarray.Dataset containing PCA data: PCs, EOFs, variance
    '''

    # predictor variable and variable_runnavg from dataset
    pred_var = xdset[pred_name]
    pred_var_ra = xdset['{0}_runavg'.format(pred_name)]

    # use datetime for indexing
    dt1 = datetime.datetime(y1,m1,1)
    dt2 = datetime.datetime(y2+1,m2,28)

    # use data inside timeframe
    data_ss = pred_var.loc[:,:,dt1:dt2]
    data_ss_ra = pred_var_ra.loc[:,:,dt1:dt2]

    # Removing the running mean of monthly mean sea levels, this gives us a
    # time series and spatial distribution of ANOMALIES in sea surface temp
    data_anom = data_ss - data_ss_ra

    # Getting an average across all Latitudes for each Longitude in the bound at each instance in time
    data_avg_lat = data_anom.mean(dim='latitude')

    # we need to reshape to collapse 12 months of data to a single vector
    nlon = data_avg_lat.longitude.shape[0]
    ntime = data_avg_lat.time.shape[0]
    hovmoller=xr.DataArray(
        np.reshape(data_avg_lat.values, (12*nlon, ntime/12), order='F'))
    hovmoller = hovmoller.transpose()

    # mean and standard deviation
    var_anom_mean = hovmoller.mean(axis=0)
    var_anom_std = hovmoller.std(axis=0)

    # Ok, so we're removing those means, and normalizing by the standard
    # deviation at anomaly.  This gives a matrix with rows = time (observations)
    # and columns = longitude (locations)
    nk_m = np.kron(np.ones((y2-y1+1,1)), var_anom_mean)
    nk_s = np.kron(np.ones((y2-y1+1,1)), var_anom_std)
    var_anom_demean = (hovmoller - nk_m)/nk_s

    # principal components analysis
    ipca = PCA(n_components=var_anom_demean.shape[0])
    PCs = ipca.fit_transform(var_anom_demean)

    print 'Principal Components Analysis COMPLETE'
    return xr.Dataset(
        {
            'PCs': (('n_components', 'n_components'), PCs),
            'EOFs': (('n_components','n_features'), ipca.components_),
            'variance': (('n_components',), ipca.explained_variance_),
        },

        attrs = {
        }
    )

