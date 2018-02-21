#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
#np.set_printoptions(threshold=np.nan)
from collections import OrderedDict
from sklearn import linear_model
from datetime import datetime, date, timedelta
import xarray

class ALR_ENV(object):
    'AutoRegressive Logistic Enveloper'

    def __init__(self, evbmus_values, evbmus_time, cluster_size):

        # evbmus series
        self.evbmus_values = evbmus_values
        self.evbmus_time = evbmus_time

        # cluster data
        self.cluster_size = cluster_size

        # ALR model core
        self.ALR_model = None

        # ALR terms
        self.d_terms_settings = {}
        self.terms_fit = {}

    def SetFittingTerms(self, d_terms_settings):
        'Set terms settings that will be used for fitting'

        # default settings used for ALR terms
        default_settings = {
            'mk_order'  : 0,
            'constant' : False,
            'long_term' : False,
            'seasonality': (False, []),
            'covariates': (False, [])
        }

        # join user and default input
        for k in default_settings.keys():
            if k not in d_terms_settings:
                d_terms_settings[k] = default_settings[k]

        # generate ALR terms
        bmus = self.evbmus_values
        time = self.evbmus_time
        cluster_size = self.cluster_size

        self.terms_fit = self.GenerateALRTerms(d_terms_settings, bmus, time,
                                               cluster_size)

        # store data
        self.mk_order = d_terms_settings['mk_order']
        self.d_terms_settings = d_terms_settings

    def GenerateALRTerms(self, d_terms_settings, bmus, time, cluster_size):
        'Generate ALR terms from user terms settings'

        # terms stored at OrderedDict
        terms = OrderedDict()

        # constant term
        if d_terms_settings['constant']:
            terms['constant'] = np.ones((bmus.size, 1))

        # time term (use custom time array with year decimals)
        if d_terms_settings['long_term']:
            terms['long_term'] = np.ones((bmus.size, 1))
            terms['long_term'][:,0] = self.GetFracYears(time)

        # seasonality term
        if d_terms_settings['seasonality'][0]:
            amplitudes  = d_terms_settings['seasonality'][1]
            time_yfrac = self.GetFracYears(time)

            temp_seas = np.zeros((len(time_yfrac), 2*len(amplitudes)))
            c = 0
            for a in amplitudes:
                temp_seas [:,c]   = np.cos(a * np.pi * time_yfrac)
                temp_seas [:,c+1] = np.sin(a * np.pi * time_yfrac)
                c+=2

            terms['seasonality'] = temp_seas

        # Covariables term (normalized)
        if d_terms_settings['covariates'][0]:
            cov_norm = d_terms_settings['covariates'][1]
            for i in range(cov_norm.shape[1]):
                terms['cov_{0}'.format(i+1)] = np.transpose(
                    np.asmatrix(cov_norm[:,i]))

        # markov term
        # TODO: COMPARAR MARKOVs CON JAAA ALR PARA COVARS
        if d_terms_settings['mk_order'] > 0:

            # dummi for markov chain
            def dummi(csize):
                D = np.ones((csize-1, csize)) * -1
                for i in range(csize-1):
                    D[i, csize-1-i] = csize-i-1
                    D[i, csize-1+1-i:] = 0
                return D
            dum = dummi(cluster_size)

            # solve markov order N
            mk_order = d_terms_settings['mk_order']
            for i in range(mk_order):
                Z = np.zeros((bmus.size, cluster_size-1))
                for indz in range(bmus.size-i-1):
                    Z[indz+i+1,0:] = np.squeeze(dum[0:,bmus[indz]-1])
                terms['markov_{0}'.format(i+1)] = Z

        return terms

    def GetFracYears(self, time):
        'Returns time in custom year decimal format'

        if isinstance(time[0], xarray.core.dataarray.DataArray):
            y0 = time[0].dt.year
            m0 = time[0].dt.month
            d0 = time[0].dt.day
            y1 = time[-1].dt.year
            m1 = time[-1].dt.month
            d1 = time[-1].dt.day
        elif isinstance(time[0], date):
            y0 = time[0].year
            m0 = time[0].month
            d0 = time[0].day
            y1 = time[-1].year
            m1 = time[-1].month
            d1 = time[-1].day
        else:
            # TODO raise error
            print 'GetFracYears time not recognized'
            import sys; sys.exit()

        # year fractions
        d0 = date(y0, m0, d0)
        d1 = date(y1, m1, d1)
        delta = d1 - d0

        return  np.array(range(delta.days+1))/365.25

    def FitModel(self):
        'Fits ARL model using terms_fit'

        # get fitting data
        evbmus = self.evbmus_values
        terms = self.terms_fit
        tmodl = np.concatenate(terms.values(), axis=1)

        # fit model
        start_time = time.time()
        self.ALR_model = linear_model.LogisticRegression(
            penalty='l2', C=1e5, fit_intercept=False)
        self.ALR_model.fit(tmodl, evbmus)
        elapsed_time = time.time() - start_time
        print "Optimization done in {0:.2f} seconds".format(elapsed_time)

        predprob = self.ALR_model.predict_proba(tmodl)
        print predprob

    def Simulate(self, num_sims, sim_start_y, sim_end_y, sim_freq, sim_covars_T=None):
        'Launch ARL model simulations'

        # get needed data
        evbmus_values = self.evbmus_values
        mk_order = self.mk_order

        # generate simulation date list
        if sim_freq == '1d':
            d1 = date(sim_start_y, 1, 1)
            d2 = date(sim_end_y, 1, 1)
            tdelta = d2 - d1
            list_sim_dates = [d1+timedelta(days=i) for i in range(tdelta.days)]
        elif sim_freq == '1y':
            list_sim_dates = [date(x,1,1) for x in
                              range(sim_start_y,sim_end_y+1)]

        # start simulations
        evbmus_sims = np.zeros((len(list_sim_dates), num_sims))
        for n in range(num_sims):
            print 'simulation num. {0}'.format(n+1)
            evbmus = evbmus_values[-mk_order:]

            # TODO: SACAF EL IF DEL BUCLE, PORQUE LO RALENTIZA
            for i in range(len(list_sim_dates) - mk_order):

                # handle optional covars
                if self.d_terms_settings['covariates'][0]:
                    sim_covars_evbmus = sim_covars_T[i : i + mk_order +1]
                    sim_cov_norm = (
                        sim_covars_evbmus - sim_covars_T.mean(axis=0)
                    ) / sim_covars_T.std(axis=0)

                    self.d_terms_settings['covariates']=(
                        True,
                        sim_cov_norm
                    )

                # generate time step ALR terms
                terms_i = self.GenerateALRTerms(
                    self.d_terms_settings,
                    np.append(evbmus[ i : i + mk_order], 0),
                    list_sim_dates[i : i + mk_order + 1],
                    self.cluster_size)

                # Event sequence simulation  
                prob = self.ALR_model.predict_proba(np.concatenate(terms_i.values(),axis=1))
                probTrans = np.cumsum(prob[-1,:])
                evbmus = np.append(evbmus, np.where(probTrans>np.random.rand())[0][0]+1)
                print i

            evbmus_sims[:,n] = evbmus

            # Probabilities in the nsims simulations
            evbmus_prob = np.zeros((evbmus_sims.shape[0], self.cluster_size))
            for i in range(evbmus_sims.shape[0]):
                for j in range(self.cluster_size):
                    evbmus_prob[i, j] = len(np.argwhere(evbmus_sims[i,:]==j+1))/float(num_sims)

        evbmus_probcum = np.cumsum(evbmus_prob, axis=1)

        return evbmus_sims, evbmus_probcum, list_sim_dates

