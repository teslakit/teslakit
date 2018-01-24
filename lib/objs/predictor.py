#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import xarray as xr

from lib.custom_stats import running_mean

class WeatherPredictor(object):
    'Predictor por Annual Weather Types methodology'

    def __init__(self, p_save):

        self.p_save = p_save

        self.data_set = None     # xr.Dataset
        self.name_pred = None     # name of the predictor variable (SST, MSL, ...)

        # TODO: utilizar el dataset para guardarlo todo
        self.PCA = None          # Principal components analysis: EOFs, PCs, varia

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

        # TODO: guardar output en xr.dataset o similar

        self.PCA = None



