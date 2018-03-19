#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op

from lib.objs.predictor import WeatherPredictor as WPred


class Project(object):
    'Tesla-Kit project framework'

    def __init__(self, p_case):

        # path control
        self.p_dict = self.GeneratePaths(p_case)


    def GeneratePaths(p_case):
        'Generates case paths template'

        return {
            'case': p_case,
            'predictor': '',
            'pred_PCA': op.join(p_case, 'xds_PCA.nc'),
            'pred_AWT': op.join(p_case, 'xds_AWT.nc'),
            'pred_AWT_ARLsim': op.join(p_case, 'xds_AWT_ALRsim.nc'),
        }

    def Generate_AWT(wp):
        '''
        Principal Components Analysis
        KMA Classification
        Annual Autoregressive Logistic Regression
        '''

        # Load Weather predictor (SST)
        wp = WPred(p_predictor)
        # we can generate a predictor using an xarray.dataset
        # with coords: longitude, latitude, time


        # Principal Components Analysis
        y1 = 1880
        yN = 2016
        m1 = 6
        mN = 5
        xds_pca = wpred.CalcPCA(y1, yN, m1, mN)
        xds_pca.to_netcdf(self.p_dict['pred_PCA'])


        # KMA Classification 
        num_clusters = 6
        num_reps = 2000
        repres = 0.95

        # TODO: ACABAR COPULAS DENTRO
        xds_AWT = ClassificationKMA(
            xds_pca, num_clusters, num_reps, repres)
        xds_pca.to_netcdf(self.p_dict['pred_AWT'])


        ## Autoregressive Logistic Regression

        xds_bmus_fit = xr.Dataset(
            {
                'bmus':(('time',), xds_AWT.bmus),
            },
            coords = {'time': xds_AWT.time.values}
        ).bmus

        num_wts = 6
        ALRE = ALR_ENV(xds_bmus_fit, num_wts)

        # ALR terms
        d_terms_settings = {
            'mk_order'  : 1,
            'constant' : True,
            'long_term' : False,
            'seasonality': (False, []),
        }

        ALRE.SetFittingTerms(d_terms_settings)

        # ALR model fitting
        ALRE.FitModel()

        # ALR model simulations 
        sim_num = 10
        year_sim1 = 1700
        year_sim2 = 2700

        dates_sim = [
            datetime(x,1,1) for x in range(year_sim1,year_sim2+1)]

        evbmus_sim, evbmus_probcum = ALRE.Simulate(
            sim_num, dates_sim)

        # TODO: CAMBIAR EL OUTOUT DE SIMULATE A UN DATASET XARRAY
        # guardarlo en path pred_AWT_ARLsim

