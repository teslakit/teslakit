#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import os.path as op
from datetime import datetime, timedelta

from lib.custom_stats import ClassificationKMA
from lib.objs.alr_enveloper import ALR_ENV
from lib.mjo import GetMJOCategories, DownloadMJO
from lib.predictor import CalcPCA_Annual_latavg as CalcPCA
from lib.custom_plot import Plot_MJOphases, Plot_MJOCategories


class Project(object):
    'Tesla-Kit project framework'

    def __init__(self, p_case):

        # path control
        self.p_dict = self.GeneratePaths(p_case)

    def GeneratePaths(self, p_case):
        'Generates case paths template'

        return {
            'case': p_case,

            'predictor': op.join(p_case, 'sst_1985_2017.nc'),
            'pred_PCA': op.join(p_case, 'xds_PCA.nc'),
            'pred_AWT': op.join(p_case, 'xds_AWT.nc'),
            'pred_AWT_ALR_sim': op.join(p_case, 'xds_AWT_ALR_sim.nc'),

            'mjo_hist': op.join(p_case, 'xds_MJO_hist.nc'),
            'mjo_ALR_sim': op.join(p_case, 'xds_MJO_ALR_sim.nc'),
        }

    def Generate_Predictor(p_pred):
        '''
        Generate predictor variable
        '''
        # TODO
        # we can generate a predictor using an xarray.dataset
        # with coords: longitude, latitude, time
        pass

    def Download_MJO(self):
        'Download historical MJO data'

        y1 = '1979-01-01'  # remove nans
        DownloadMJO(self.p_dict['mjo_hist'], init_year=y1, log=True)

    def Generate_AWT_simulation(self):
        '''
        Generate Annual Weather Types simulation:

        Principal Components Analysis
        KMA Classification
        Annual Autoregressive Logistic Regression
        '''

        # Load Weather predictor (SST)
        xds_pred = xr.open_dataset(self.p_dict['predictor'])
        lon_pred = xds_pred.longitude.values


        ## Principal Components Analysis
        pred_name = 'SST'
        y1 = 1880
        yN = 2016
        m1 = 6
        mN = 5

        xds_PCA = CalcPCA(xds_pred, pred_name, y1, yN, m1, mN)

        xds_PCA.to_netcdf(self.p_dict['pred_PCA'], 'w')


        ## KMA Classification 
        num_clusters = 6
        repres = 0.95

        # TODO: ACABAR COPULAS DENTRO
        xds_AWT = ClassificationKMA(
            xds_PCA, num_clusters, repres)

        # add yearly time data to xds_AWT
        time_yearly = [datetime(x,1,1) for x in range(y1,yN+1)]
        xds_AWT['time']=(('n_pcacomp'), time_yearly)

        xds_AWT.to_netcdf(self.p_dict['pred_AWT'], 'w')


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

        xds_alr = ALRE.Simulate(sim_num, dates_sim)
        xds_alr.to_netcdf(self.p_dict['pred_AWT_ALR_sim'], 'w')

    def Generate_MJO_simulation(self):
        '''
        Generate MJO simulation:

        Uses MJO components rmm1 and rmm2
        Classification in categories
        Daily Autoregressive Logistic Regression
        '''

        # Load MJO data (previously downloaded)
        xds_mjo_hist = xr.open_dataset(self.p_dict['mjo_hist'])

        # Calculate MJO categories (25 used) 
        rmm1 = xds_mjo_hist['rmm1']
        rmm2 = xds_mjo_hist['rmm2']
        phase = xds_mjo_hist['phase']

        categ, d_rmm_categ = GetMJOCategories(rmm1, rmm2, phase)
        xds_mjo_hist['categ'] = (('time',), categ)


        # Autoregressive logistic enveloper
        num_categs  = 25
        xds_bmus_fit = xds_mjo_hist.categ
        ALRE = ALR_ENV(xds_bmus_fit, num_categs)

        # ALR terms
        d_terms_settings = {
            'mk_order'  : 3,
            'constant' : True,
            'seasonality': (True, [2,4,8]),
        }
        ALRE.SetFittingTerms(d_terms_settings)

        # ALR model fitting
        ALRE.FitModel()

        # ALR model simulations 
        sim_num = 1  # only one simulation for mjo daily
        sim_years = 15

        # simulation dates
        d1 = datetime(1900,1,1)
        d2 = datetime(d1.year+sim_years, d1.month, d1.day)
        dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

        # launch simulation
        xds_alr = ALRE.Simulate(sim_num, dates_sim)
        xds_alr.to_netcdf(self.p_dict['mjo_ALR_sim'], 'w')


        # Generate mjo_sim components (rmm1,rmm2) using random stored mjo for each categ
        evbmus_sim = np.squeeze(xds_alr.evbmus_sims.values)
        mjo_sim_rmm1 = np.empty((len(evbmus_sim),1)) * np.nan
        mjo_sim_rmm2 = np.empty((len(evbmus_sim),1)) * np.nan
        for c, m in enumerate(evbmus_sim):
            options = d_rmm_categ['cat_{0}'.format(int(m))]
            r = np.random.randint(options.shape[0])
            mjo_sim_rmm1[c] = options[r,0]
            mjo_sim_rmm2[c] = options[r,1]

        # Store mjo simulated data
        xds_mjo_sim = xr.Dataset(
            {
                'categ':(('time',), evbmus_sim),
                'rmm1':(('time',), np.squeeze(mjo_sim_rmm1)),
                'rmm2':(('time',), np.squeeze(mjo_sim_rmm2)),
            },

            coords = {
                'time' : [np.datetime64(d) for d in dates_sim],
            },

            attrs = {
                'name': 'MJO simulated with ALR'
            }
        )
        xds_mjo_sim.to_netcdf(self.p_dict['mjo_ALR_sim'])

    def Plot_MJO_historical(self):

        # Load MJO data
        xds_mjo_hist = xr.open_dataset(self.p_dict['mjo_hist'])

        rmm1 = xds_mjo_hist['rmm1']
        rmm2 = xds_mjo_hist['rmm2']
        phase = xds_mjo_hist['phase']
        categ, _ = GetMJOCategories(rmm1, rmm2, phase)

        # plot MJO data
        Plot_MJOphases(rmm1, rmm2, phase)

        # plot MJO categories
        Plot_MJOCategories(rmm1, rmm2, categ)


