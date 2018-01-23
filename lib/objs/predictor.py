#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr

from lib.custom_stats import running_mean

class WeatherPredictor(object):
    'Predictor por Annual Weather Types methodology'

    def __init__(self, data_pred, var_name):
        # TODO: LIGARLO A UN ARCHIVO 

        self.data = data_pred     # xr.DataArray (lon, lat, time)
        self.var_name = var_name  # name of the predictor variable (SST, MSL, ...)

        self.data_runavg = None   # running average by months

        self.PCA = None           # Principal components analysis: EOFs, PCs, varia

    def CalcRunningMean(self, window):
        'Calculates running mean same as Dylan methodology'

        tempdata_runavg = np.empty(self.data.shape)

        for lon in self.data.longitude.values:
            for lat in self.data.latitude.values:
                for mn in range(1, 13):

                    # indexes
                    ix_lon = np.where(self.data.longitude == lon)
                    ix_lat = np.where(self.data.latitude == lat)
                    ix_mn = np.where(self.data['time.month'] == mn)

                    # point running average
                    time_mn = self.data.time[ix_mn]
                    data_pnt = self.data.loc[lon, lat, time_mn]
                    data_pnt = self.data[lon, lat, time_mn]

                    tempdata_runavg[ix_lon[0], ix_lat[0], ix_mn[0]] = running_mean(data_pnt.values, 5)

        # store data
        self.data_runavg = xr.DataArray(
            tempdata_runavg,
            coords=[lon, lat, time],
            dims=['longitude', 'latitude', 'time'])

    def CalcPCA(self, y1, y2, m1, m2):
        'Principal component analysis'

        # TODO: guardar output en xr.dataset o similar

        self.PCA = None



