#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import xarray as xr
import datetime
from sklearn.decomposition import PCA

from lib.custom_stats import running_mean

class WeatherPredictor(object):
    'Predictor por Annual Weather Types methodology'

    # TODO: INCORPORAR PLOTEOS CON MATPLOTLIB (interactivo? plt.show()?) 

    def __init__(self, p_save):

        self.p_save = p_save

        self.data_set = None     # xr.Dataset
        self.name_pred = None     # name of the predictor variable (SST, MSL, ...)

        # Load data from file if it exists
        self.LoadData()

    def SetData(self, data_pred):
        'Sets WeatherPredictor data from an xr.DataArray'

        self.data_set = xr.Dataset({'predictor': data_pred})
        self.name_pred = data_pred.name

    def LoadData(self, p_load=None):
        'Load WeatherPredictor data from netcdf file'

        # if not file given, load from default savefile
        if not p_load:
            p_load = self.p_save

        # load data
        if op.isfile(p_load):
            self.data_set = xr.open_dataset(p_load)
            print 'Predictor Dataset loaded from {0}'.format(p_load)

    def SaveData(self, p_save=None):
        'Save WeatherPredictor data to netcdf file'

        # if not file given, save to default savefile
        if not p_save:
            p_save = self.p_save

        # save data
        if op.isfile(p_save):
            os.remove(p_save)
        self.data_set.to_netcdf(p_save, 'w')
        print 'Predictor Dataset saved at {0}'.format(p_save)

    def CalcRunningMean(self, window):
        'Calculates running mean same as Dylan methodology'
        # TODO: MUY LENTO, OPTIMIZAR

        tempdata_runavg = np.empty(self.data_set['predictor'].shape)

        for lon in self.data_set.longitude.values:
            for lat in self.data_set.latitude.values:
                for mn in range(1, 13):

                    # indexes
                    ix_lon = np.where(self.data_set.longitude == lon)
                    ix_lat = np.where(self.data_set.latitude == lat)
                    ix_mnt = np.where(self.data_set['time.month'] == mn)

                    # point running average
                    time_mnt = self.data_set.time[ix_mnt]
                    data_pnt = self.data_set['predictor'].loc[lon, lat, time_mnt]


                    tempdata_runavg[ix_lon[0], ix_lat[0], ix_mnt[0]] = running_mean(data_pnt.values, 5)

        # store data
        self.data_set['predictor_runavg'] = (('longitude', 'latitude', 'time'),
                                             tempdata_runavg)

    def CalcPCA(self, y1, y2, m1, m2):
        'Principal component analysis'

        # use datetime for indexing
        dt1 = datetime.datetime(y1,m1,1)
        dt2 = datetime.datetime(y2+1,m2,28)

        # use data inside timeframe
        data_ss = self.data_set['predictor'].loc[:,:,dt1:dt2]
        data_ss_ra = self.data_set['predictor_runavg'].loc[:,:,dt1:dt2]

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

        EOFs = xr.DataArray(ipca.components_)
        variance = xr.DataArray(ipca.explained_variance_)

        print 'Principal Components Analysis COMPLETE'
        print ipca

        return {
            'PCs': PCs,
            'EOFs': EOFs,
            'variance': variance
        }

