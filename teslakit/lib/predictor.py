#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import datetime
from sklearn.decomposition import PCA
from matplotlib import path

# tk libs
from lib.custom_stats import running_mean
from lib.util.terminal import printProgressBar as pb


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

def spatial_gradient(xdset, var_name):
    '''
    Calculate spatial gradient

    xdset:
        (longitude, latitude, time), var_name

    returns xdset with new variable "var_name_gradient"
    '''

    var_grad = np.zeros(xdset[var_name].shape)

    Mx = len(xdset.longitude)
    My = len(xdset.latitude)
    lat = xdset.latitude.values

    for it in range(len(xdset.time)):
        var_val = xdset[var_name].isel(time=it).values

        for i in range(1, Mx-1):
            for j in range(1, My-1):
                phi = np.pi*np.abs(lat[j])/180.0
                dpx1 = (var_val[j,i]   - var_val[j,i-1]) / np.cos(phi)
                dpx2 = (var_val[j,i+1] - var_val[j,i])   / np.cos(phi)
                dpy1 = (var_val[j,i]   - var_val[j-1,i])
                dpy2 = (var_val[j+1,i] - var_val[j,i])

                var_grad[it, j, i] = (dpx1**2+dpx2**2)/2 + (dpy1**2+dpy2**2)/2

    # store gradient
    xdset['{0}_gradient'.format(var_name)]= (
        ('time', 'latitude', 'longitude'), var_grad)

    return xdset

def mask_from_poly(xdset, ls_poly):
    '''
    Generate mask from list of tuples (lon, lat)

    xdset dimensions:
        (longitude, latitude, )

    returns xdset with new variable "mask"
    '''

    lon = xdset.longitude.values
    lat = xdset.latitude.values
    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mask = np.zeros(mesh_lat.shape)

    mesh_points = np.array(
        [mesh_lon.flatten(), mesh_lat.flatten()]
    ).T

    for pol in ls_poly:
        p = path.Path(pol)
        inside = np.array(p.contains_points(mesh_points))
        inmesh = np.reshape(inside, mask.shape)
        mask[inmesh] = 1

    xdset['mask']=(('latitude','longitude'), mask.T)

    return xdset

def dynamic_estela_predictor(xdset, var_name, estela_D):
    '''
    Generate dynamic predictor using estela

    xdset:
        (longitude, latitude, time), var_name

    returns xdset with new variables:
        var_name_comp, var_name_gradient_comp
    '''
    first_day = int(np.floor(np.nanmax(estela_D)))+1

    var_comp = np.ones(xdset[var_name].shape) * np.nan
    var_grd_comp = np.ones(xdset[var_name].shape) * np.nan

     # TODO: ARREGLAR 

    for lat in range(len(xdset.latitude)):
        for lon in range(len(xdset.longitude)):
            ed = estela_D[lat, lon]
            if not np.isnan(ed):
                t_indexes = np.arange(
                    first_day, len(xdset.time)) - np.int(ed)
                xdselec= xdset.isel(
                    time = t_indexes,
                    latitude=lat,
                    longitude=lon)

                var_comp[t_indexes,lat,lon] = xdselec[var_name].values
                var_grd_comp[t_indexes,lat,lon] = \
                xdselec['{0}_gradient'.format(var_name)].values

    # store estela predictor
    xdset['{0}_comp'.format(var_name)]= (
        ('time', 'latitude', 'longitude'), var_comp)
    xdset['{0}_gradient_comp'.format(var_name)]= (
        ('time', 'latitude', 'longitude'), var_grd_comp)

    return xdset

def CalcPCA_EstelaPred():
    # TODO
    return None
