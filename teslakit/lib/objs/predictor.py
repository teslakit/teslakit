#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# tk libs
from lib.estela import dynamic_estela_predictor
from lib.PCA import CalcPCA_EstelaPred
from lib.KMA import KMA_regression_guided
from lib.KMA import SimpleMultivariateRegressionModel as SMRM
from lib.intradaily import Calculate_Hydrographs
from lib.plotting.EOFs import Plot_EOFs_EstelaPred
from lib.plotting.KMA import Plot_KMArg_clusters_datamean


class Predictor(object):
    '''
    tesla-kit custom dataset handler

    used for 3D dataset (lon,lat,time) and related
    statistical classification calculations and figures.
    '''

    def __init__(self, p_store):

        # file paths
        self.p_store = p_store
        self.p_data = op.join(p_store, 'data.nc')
        self.p_pca = op.join(p_store, 'pca.nc')
        self.p_kma = op.join(p_store, 'kma.nc')
        self.p_plots = op.join(p_store, 'figs')

        # data (xarray.Dataset)
        self.data = None
        self.PCA = None
        self.KMA = None

    def Load(self):
        if op.isfile(self.p_data):
            self.data = xr.open_dataset(self.p_data)
        if op.isfile(self.p_pca):
            self.PCA = xr.open_dataset(self.p_pca)
        if op.isfile(self.p_kma):
            self.KMA = xr.open_dataset(self.p_kma)

    def Save(self):
        try:
            os.makedirs(self.p_store)
        except:
            pass
        if self.data:
            if op.isfile(self.p_data): os.remove(self.p_data)
            self.data.to_netcdf(self.p_data,'w')
        if self.PCA:
            if op.isfile(self.p_pca): os.remove(self.p_pca)
            self.PCA.to_netcdf(self.p_pca,'w')
        if self.KMA:
            if op.isfile(self.p_kma): os.remove(self.p_kma)
            self.KMA.to_netcdf(self.p_kma,'w')

    def Calc_PCA_EstelaPred(self, var_name, xds_estela):
        'Principal components analysis using estela predictor'

        # generate estela predictor
        xds_estela_pred = dynamic_estela_predictor(
            self.data, var_name, xds_estela)

        # Calculate PCA
        self.PCA = CalcPCA_EstelaPred(
            xds_estela_pred, var_name)

    def Calc_KMA_regressionguided(
        self, num_clusters, xds_waves, waves_vars, alpha):
        'KMA regression guided with waves data'

        # we have to miss some days of data due to ESTELA
        tcut = self.PCA.pred_time.values[:]

        # calculate regresion model between predictand and predictor
        xds_waves = xds_waves.sel(time = slice(tcut[0], tcut[-1]))
        xds_Yregres = SMRM(self.PCA, xds_waves, waves_vars)

        # classification: KMA regresion guided
        repres = 0.95
        self.KMA = KMA_regression_guided(
            self.PCA, xds_Yregres, num_clusters, repres, alpha)

        # store time array with KMA
        self.KMA['time'] = (('n_components',), self.PCA.pred_time.values[:])

    def Calc_MU_TAU_Hydrographs(self, xds_WAVES):
        '''
        Calculates TWL hydrographs

        returns list of xarray.Dataset with TWL hydrographs MU,TAU arrays for each WT
        '''

        # get sorted bmus from kma
        xds_BMUS = xr.Dataset(
            {'bmus':(('time', self.KMA.sorted_bmus.values[:]))},
            coords = {'time': self.KMA.time.values[:]}
        )

        # Calculate hydrographs for each WT
        _, l_xds_MUTAU = Calculate_Hydrographs(xds_BMUS, xds_WAVES)

        return l_xds_MUTAU

    def Mod_KMA_AddStorms(self, storm_dates, storm_categories):
        '''
        Modify KMA bmus series adding storm category (6 new groups)
        '''

        n_clusters = len(self.KMA.n_clusters.values[:])
        kma_dates = self.PCA.pred_time.values[:]
        bmus_storms = self.KMA.sorted_bmus.copy()  # deep copy

        for sd, sc in zip(storm_dates, storm_categories):
            pos_date = np.where(kma_dates==sd)[0]
            if pos_date:
                bmus_storms[pos_date[0]] = n_clusters + sc + 1

        # copy kma and add bmus_storms
        self.KMA['sorted_bmus_storms'] = (('n_components',), bmus_storms)

        # store changes
        if op.isfile(self.p_kma): os.remove(self.p_kma)
        self.KMA.to_netcdf(self.p_kma,'w')

    def Plot_EOFs_EstelaPred(self, n_plot, show=False):
        'Plot EOFs generated in PCA_EstelaPred'

        if show:
            p_export = None
        else:
            p_export = op.join(self.p_store, 'EOFs_EP')

        Plot_EOFs_EstelaPred(self.PCA, n_plot, p_export)

    def Plot_KMArg_clusters_datamean(self, var_name, show=False, mask_name=None):
        '''
        Plot KMA clusters generated in PCA_EstelaPred
        uses database means at cluster location (bmus corrected)
        '''

        if show:
            p_export = None
        else:
            p_export = op.join(
                self.p_store,
                'KMA_RG_clusters_datamean_{0}.png'.format(var_name))

        bmus = self.KMA['sorted_bmus'].values
        var_data = self.data[var_name]

        if mask_name:
            var_data = var_data.where(self.data[mask_name]==1)

        Plot_KMArg_clusters_datamean(var_data, bmus, p_export)

