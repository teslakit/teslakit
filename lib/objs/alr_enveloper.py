#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
#np.set_printoptions(threshold=np.nan)
from collections import OrderedDict
from sklearn import linear_model
import scipy.stats as stat
from datetime import datetime, date, timedelta
import xarray
import pickle
from lib.util.terminal import printProgressBar as pb

# TODO: introducir modelo de statsmodels 
# TODO: ajustar las ejecuciones anuales a los cambios

class ALR_ENV(object):
    'AutoRegressive Logistic Enveloper'

    def __init__(self, xds_bmus_fit, cluster_size):

        # evbmus series
        self.evbmus_values = xds_bmus_fit.values
        self.evbmus_time = xds_bmus_fit.time.values

        # cluster data
        self.cluster_size = cluster_size

        # ALR terms
        self.d_terms_settings = {}
        self.terms_fit = {}

        # ALR model core
        self.model = None

        # ALR model auxiliar vars
        self.p_values = None

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
                                               cluster_size, time2yfrac=True)

        # store data
        self.mk_order = d_terms_settings['mk_order']
        self.d_terms_settings = d_terms_settings

    def GenerateALRTerms(self, d_terms_settings, bmus, time, cluster_size,
                         time2yfrac=False):
        'Generate ALR terms from user terms settings'

        # terms stored at OrderedDict
        terms = OrderedDict()

        # time options (time has to bee yearly fraction)
        if time2yfrac:
            time_yfrac = self.GetFracYears(time)
        else:
            time_yfrac = time

        # constant term
        if d_terms_settings['constant']:
            terms['constant'] = np.ones((bmus.size, 1))

        # time term (use custom time array with year decimals)
        if d_terms_settings['long_term']:
            terms['long_term'] = np.ones((bmus.size, 1))
            terms['long_term'][:,0] = time_yfrac

        # seasonality term
        if d_terms_settings['seasonality'][0]:
            amplitudes  = d_terms_settings['seasonality'][1]
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

        # get start/end data. resolution day
        y0 = time[0].year
        m0 = time[0].month
        d0 = time[0].day
        y1 = time[-1].year
        m1 = time[-1].month
        d1 = time[-1].day

        # start "year cicle" at 01/01 
        d_y0 = date(y0, 01, 01)

        # time array
        d_0 = date(y0, m0, d0)
        d_1 = date(y1, m1, d1)

        # year_decimal from year start to d1
        delta_y0 = d_1 - d_y0
        y_fraq_y0 = np.array(range(delta_y0.days+1))/365.25

        # cut year_decimal from d_0 
        i0 = (d_0-d_y0).days
        y_fraq = y_fraq_y0[i0:]

        return y_fraq

    def FitModel(self):
        'Fits ARL model using sklearn'

        # get fitting data
        X = np.concatenate(self.terms_fit.values(), axis=1)
        y = self.evbmus_values

        # fit model
        print "\nFitting autoregressive logistic model..."
        start_time = time.time()

        self.model = linear_model.LogisticRegression(
            penalty='l2', C=1e5, fit_intercept=False)
        self.model.fit(X, y)

        elapsed_time = time.time() - start_time
        print "Optimization done in {0:.2f} seconds\n".format(elapsed_time)

        # TODO
        # Get p-values from sklearn 

    def SaveModel(self, p_save):
        'Saves fitted model for future use'

        pickle.dump(self.model, open(p_save, 'wb'))
        print 'ALR model saved at {0}'.format(p_save)

    def LoadModel(self, p_load):
        'Load fitted model'

        self.model = pickle.load(open(p_load, 'rb'))
        print 'ALR model loaded from {0}'.format(p_load)

    def Simulate(self, num_sims, list_sim_dates, sim_covars_T=None):
        'Launch ARL model simulations'
        # TODO: CAMBIAR TESTS PARA LAS SIMULACIONES ANUALES

        # get needed data
        evbmus_values = self.evbmus_values
        mk_order = self.mk_order

        # generate time yearly fractional array
        time_yfrac = self.GetFracYears(list_sim_dates)

        # start simulations
        print "\nLaunching simulations...\n"
        evbmus_sims = np.zeros((len(time_yfrac), num_sims))
        for n in range(num_sims):

            #print 'Sim. Num. {0}'.format(n+1)
            evbmus = evbmus_values[1:mk_order+1] # TODO: arreglado, comentar
            for i in range(len(time_yfrac) - mk_order):

                # handle optional covars
                # TODO: optimizar manejo covars
                # seria perfecto no tener que alterar self.d_term_settings aqui
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
                    time_yfrac[i : i + mk_order + 1],
                    self.cluster_size, time2yfrac=False)

                # Event sequence simulation  
                prob = self.model.predict_proba(np.concatenate(terms_i.values(),axis=1))
                probTrans = np.cumsum(prob[-1,:])
                evbmus = np.append(evbmus, np.where(probTrans>np.random.rand())[0][0]+1)

                # progress bar
                pb(i + 1, len(time_yfrac),
                   prefix = 'Sim. Num. {0}'.format(n+1),
                   suffix = 'Complete', length = 50)

            evbmus_sims[:,n] = evbmus

            # progress bar
            pb(len(time_yfrac), len(time_yfrac),
                prefix = 'Sim. Num. {0}'.format(n+1),
                suffix = 'Complete', length = 50)

            # Probabilities in the nsims simulations
            evbmus_prob = np.zeros((evbmus_sims.shape[0], self.cluster_size))
            for i in range(evbmus_sims.shape[0]):
                for j in range(self.cluster_size):
                    evbmus_prob[i, j] = len(np.argwhere(evbmus_sims[i,:]==j+1))/float(num_sims)

        evbmus_probcum = np.cumsum(evbmus_prob, axis=1)

        return evbmus_sims, evbmus_probcum

