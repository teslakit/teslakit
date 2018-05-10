#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import datetime
from sklearn.decomposition import PCA
from matplotlib import path

# tk libs
from lib.util.terminal import printProgressBar as pb


def running_mean(x, N, mode_str='mean'):
    '''
    computes a running mean (also known as moving average)
    on the elements of the vector X. It uses a window of 2*M+1 datapoints

    As always with filtering, the values of Y can be inaccurate at the
    edges. RUNMEAN(..., MODESTR) determines how the edges are treated. MODESTR can be
    one of the following strings:
      'edge'    : X is padded with first and last values along dimension
                  DIM (default)
      'zeros'   : X is padded with zeros
      'ones'    : X is padded with ones
      'mean'    : X is padded with the mean along dimension DIM

    X should not contains NaNs, yielding an all NaN result.
    '''

    # if nan in data, return nan array
    if np.isnan(x).any():
        return np.full(x.shape, np.nan)

    nn = 2*N+1

    if mode_str == 'zeros':
        x = np.insert(x, 0, np.zeros(N))
        x = np.append(x, np.zeros(N))

    elif mode_str == 'ones':
        x = np.insert(x, 0, np.ones(N))
        x = np.append(x, np.ones(N))

    elif mode_str == 'edge':
        x = np.insert(x, 0, np.ones(N)*x[0])
        x = np.append(x, np.ones(N)*x[-1])

    elif mode_str == 'mean':
        x = np.insert(x, 0, np.ones(N)*np.mean(x))
        x = np.append(x, np.ones(N)*np.mean(x))


    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[nn:] - cumsum[:-nn]) / float(nn)

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

def mask_from_poly(xdset, ls_poly, name_mask='mask'):
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

    xdset[name_mask]=(('latitude','longitude'), mask.T)

    return xdset

def dynamic_estela_predictor(xdset, var_name, estela_D):
    '''
    Generate dynamic predictor using estela

    xdset:
        (time, latitude, longitude), var_name, mask

    returns similar xarray.Dataset with variables:
        (time, latitude, longitude), var_name_comp
        (time, latitude, longitude), var_name_gradient_comp
    '''

    # first day is estela max
    first_day = int(np.floor(np.nanmax(estela_D)))+1

    # output will start at time=first_day
    shp = xdset[var_name].shape
    comp_shape = (shp[0]-first_day, shp[1], shp[2])
    var_comp = np.ones(comp_shape) * np.nan
    var_grd_comp = np.ones(comp_shape) * np.nan

    # get data using estela for each cell
    for i_lat in range(len(xdset.latitude)):
        for i_lon in range(len(xdset.longitude)):
            ed = estela_D[i_lat, i_lon]
            if not np.isnan(ed):

                # mount estela displaced time array 
                i_times = np.arange(
                    first_day, len(xdset.time)
                ) - np.int(ed)

                # select data from displaced time array positions
                xdselec = xdset.isel(
                    time = i_times,
                    latitude = i_lat,
                    longitude = i_lon)

                # get estela predictor values
                var_comp[:, i_lat, i_lon] = xdselec[var_name].values
                var_grd_comp[:, i_lat, i_lon] = xdselec['{0}_gradient'.format(var_name)].values

    # return generated estela predictor
    return xr.Dataset(
        {
            '{0}_comp'.format(var_name):(
                ('time','latitude','longitude'), var_comp),
            '{0}_gradient_comp'.format(var_name):(
                ('time','latitude','longitude'), var_grd_comp),

        },
        coords = {
            'time':xdset.time.values[first_day:],
            'latitude':xdset.latitude.values,
            'longitude':xdset.longitude.values,
        }
    )

def CalcPCA_EstelaPred(xdset, pred_name):
    '''
    Principal component analysis
    Custom for estela predictor

    xdset:
        (time, latitude, longitude), pred_name_comp | pred_name_gradient_comp

    returns a xarray.Dataset containing PCA data: PCs, EOFs, variance
    '''

    # estela predictor and estela gradient predictor
    pred_est_var = xdset['{0}_comp'.format(pred_name)]
    pred_est_grad = xdset['{0}_gradient_comp'.format(pred_name)]

    # use data inside timeframe
    dp_var = pred_est_var.values
    dp_grd = pred_est_grad.values

    # unravel and join var and grad data 
    dp_ur = np.nan * np.ones(
        (dp_var.shape[0], 2*dp_var.shape[1]*dp_var.shape[2])
    )

    # we use .T to equal matlab
    for ti in range(dp_ur.shape[0]):
        dp_ur[ti,:] = np.concatenate(
            [np.ravel(dp_var[ti].T) , np.ravel(dp_grd[ti].T)]
        )

    # remove nans from predictor    
    data_pos = ~np.isnan(dp_ur[0,:])
    clean_row = dp_ur[0, data_pos]
    dp_ur_nonan = np.nan * np.ones(
        (dp_ur.shape[0], len(clean_row))
    )
    for ti in range(dp_ur.shape[0]):
        dp_ur_nonan[ti,:] = dp_ur[ti, data_pos]


    # TODO: USAMOS DATOS MATLAB PARA EL TEST
    from lib.io.matlab import ReadMatfile
    dmat=ReadMatfile('/Users/ripollcab/Projects/TESLA-kit/source/teslakit/data/tests_estela_PCA/matlab.mat')

    dp_ur_nonan = dmat['SlpGrd']


    # standarize predictor
    pred_mean = np.mean(dp_ur_nonan, axis=0)
    pred_std = np.std(dp_ur_nonan, axis=0)
    pred_norm = (dp_ur_nonan[:,:] - pred_mean) / pred_std
    pred_norm[np.isnan(pred_norm)] = 0

    # TODO: SEPARATE CALIBRATION AND VALIDATION USING DATE
    pred_norm_cal = pred_norm
    pred_norm_val = np.array([])

    # principal components analysis
    ipca = PCA(n_components=pred_norm_cal.shape[0])
    PCs = ipca.fit_transform(pred_norm_cal)

    # return dataset
    print 'Principal Components Analysis COMPLETE'
    return xr.Dataset(
        {
            'PCs': (('n_components', 'n_components'), PCs),
            'EOFs': (('n_components','n_features'), ipca.components_),
            'variance': (('n_components',), ipca.explained_variance_),

            'pred_mean': (('n_features',), pred_mean),
            'pred_std': (('n_features',), pred_std),
        },

        attrs = {
        }
    )

