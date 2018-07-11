#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
#np.set_printoptions(threshold=np.nan)
from collections import OrderedDict
from sklearn import linear_model
import statsmodels.discrete.discrete_model as sm
import scipy.stats as stat
from datetime import datetime, date, timedelta
import xarray as xr
import pickle
import sys
import os
import os.path as op

from lib.custom_dateutils import xds2datetime as x2d
from lib.util.terminal import printProgressBar as pb
from lib.plotting.ALR import Plot_PValues, Plot_Params
from lib.plotting.ALR import Plot_Compare_Covariate, Plot_Compare_PerpYear


# fix library
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


class ALR_WRP(object):
    'AutoRegressive Logistic Enveloper'

    def __init__(self, xds_bmus_fit, cluster_size):

        # bmus dataset (vars: bmus, dims: time)
        self.xds_bmus_fit = xds_bmus_fit

        # cluster data
        self.cluster_size = cluster_size

        # ALR terms
        self.d_terms_settings = {}
        self.terms_fit = {}
        self.terms_fit_names = []

        # temporal data storage
        self.mk_order = 0
        self.cov_names = []


        # ALR model core
        self.model = None

        # config
        self.model_library = 'statsmodels'  # sklearn / statsmodels

    def SetFittingTerms(self, d_terms_settings):
        'Set terms settings that will be used for fitting'

        # default settings used for ALR terms
        default_settings = {
            'mk_order'  : 0,
            'constant' : False,
            'long_term' : False,
            'seasonality': (False, []),
            'covariates': (False, []),
            'covariates_seasonality': (False, []),
        }

        # join user and default input
        for k in default_settings.keys():
            if k not in d_terms_settings:
                d_terms_settings[k] = default_settings[k]

        # generate ALR terms
        bmus_fit = self.xds_bmus_fit.bmus.values
        time_fit = self.xds_bmus_fit.time.values
        cluster_size = self.cluster_size

        self.terms_fit, self.terms_fit_names = self.GenerateALRTerms(
            d_terms_settings, bmus_fit, time_fit, cluster_size, time2yfrac=True)

        # store data
        self.mk_order = d_terms_settings['mk_order']
        self.d_terms_settings = d_terms_settings

    def GenerateALRTerms(self, d_terms_settings, bmus, time, cluster_size,
                         time2yfrac=False):
        'Generate ALR terms from user terms settings'

        # terms stored at OrderedDict
        terms = OrderedDict()
        terms_names = []

        # time options (time has to bee yearly fraction)
        if time2yfrac:
            time_yfrac = self.GetFracYears(time)
        else:
            time_yfrac = time

        # constant term
        if d_terms_settings['constant']:
            terms['constant'] = np.ones((bmus.size, 1))
            terms_names.append('intercept')

        # time term (use custom time array with year decimals)
        if d_terms_settings['long_term']:
            terms['long_term'] = np.ones((bmus.size, 1))
            terms['long_term'][:,0] = time_yfrac
            terms_names.append('long_term')

        # seasonality term
        if d_terms_settings['seasonality'][0]:
            amplitudes  = d_terms_settings['seasonality'][1]
            temp_seas = np.zeros((len(time_yfrac), 2*len(amplitudes)))
            c = 0
            for a in amplitudes:
                temp_seas [:,c]   = np.cos(a * np.pi * time_yfrac)
                temp_seas [:,c+1] = np.sin(a * np.pi * time_yfrac)
                terms_names.append('ss_cos_{0}'.format(a))
                terms_names.append('ss_sin_{0}'.format(a))
                c+=2
            terms['seasonality'] = temp_seas

        # Covariates term
        if d_terms_settings['covariates'][0]:

            # covariates dataset (vars: cov_values, dims: time, cov_names)
            xds_cov = d_terms_settings['covariates'][1]
            cov_names = xds_cov.cov_names.values
            self.cov_names = cov_names  # storage

            # normalize covars
            if not 'cov_norm' in xds_cov.keys():
                cov_values = xds_cov.cov_values.values
                cov_norm = (cov_values - cov_values.mean(axis=0)) / cov_values.std(axis=0)
            else:
                # simulation covars are previously normalized
                cov_norm = xds_cov.cov_norm.values

            # generate covar terms
            for i in range(cov_norm.shape[1]):
                cn = cov_names[i]
                terms[cn] = np.transpose(np.asmatrix(cov_norm[:,i]))
                terms_names.append(cn)

                # Covariates seasonality
                if d_terms_settings['covariates_seasonality'][0]:
                    cov_season = d_terms_settings['covariates_seasonality'][1]

                    if cov_season[i]:
                        terms['{0}_cos'.format(cn)] = np.multiply(
                            terms[cn].T, np.cos(2*np.pi*time_yfrac)
                        ).T
                        terms['{0}_sin'.format(cn)] = np.multiply(
                            terms[cn].T, np.sin(2*np.pi*time_yfrac)
                        ).T
                        terms_names.append('{0}_cos'.format(cn))
                        terms_names.append('{0}_sin'.format(cn))

        # markov term
        if d_terms_settings['mk_order'] > 0:

            # dummi for markov chain
            def dummi(csize):
                D = np.ones((csize-1, csize)) * -1
                for i in range(csize-1):
                    D[i, csize-1-i] = csize-i-1
                    D[i, csize-1+1-i:] = 0
                return D

            def dummi_norm(csize):
                D = np.zeros((csize, csize-1))
                for i in range(csize-1):
                    D[i,i] = float((csize-i-1))/(csize-i)
                    D[i+1:,i] = -1.0/(csize-i)

                return np.transpose(np.flipud(D))

            def helmert_ints(csize, reverse=False):
                D = np.zeros((csize, csize-1))
                for i in range(csize-1):
                    D[i,i] = csize-i-1
                    D[i+1:,i] = -1.0

                if reverse:
                    return np.fliplr(np.flipud(D))
                else:
                    return D

            def helmert_norm(csize, reverse=False):
                D = np.zeros((csize, csize-1))
                for i in range(csize-1):
                    D[i,i] = float((csize-i-1))/(csize-i)
                    D[i+1:,i] = -1.0/(csize-i)

                if reverse:
                    return np.fliplr(np.flipud(D))
                else:
                    return D

            #  helmert
            dum = helmert_norm(cluster_size, reverse=True)

            # solve markov order N
            mk_order = d_terms_settings['mk_order']
            for i in range(mk_order):
                Z = np.zeros((bmus.size, cluster_size-1))
                for indz in range(bmus.size-i-1):
                    Z[indz+i+1,:] = np.squeeze(dum[bmus[indz]-1,:])

                terms['markov_{0}'.format(i+1)] = Z

                for ics in range(cluster_size-1):
                    terms_names.append(
                        'mk{0}_{1}'.format(i+1,ics+1)
                    )

        return terms, terms_names

    def GetFracYears(self, time):
        'Returns time in custom year decimal format'

        # fix np.datetime64
        if not 'year' in dir(time[0]):
            time_0 = pd.to_datetime(time[0])
            time_1 = pd.to_datetime(time[-1])
            time_d = pd.to_datetime(time[1])
        else:
            time_0 = time[0]
            time_1 = time[-1]
            time_d = time[1]

        # resolution year
        if time_d.year - time_0.year == 1:
            return range(time_1.year - time_0.year+1)

        # resolution day: get start/end data
        y0 = time_0.year
        m0 = time_0.month
        d0 = time_0.day
        y1 = time_1.year
        m1 = time_1.month
        d1 = time_1.day

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

    def FitModel(self, max_iter=1000):
        'Fits ARL model using sklearn'


        # get fitting data
        X = np.concatenate(self.terms_fit.values(), axis=1)
        y = self.xds_bmus_fit.bmus.values

        # fit model
        print "\nFitting autoregressive logistic model ..."
        start_time = time.time()

        if self.model_library == 'statsmodels':

            # mount data with pandas
            X = pd.DataFrame(X, columns=self.terms_fit_names)
            y = pd.DataFrame(y, columns=['bmus'])

            # TODO: CAPTURAR LA EVOLUCION DE LOS VALORES DURANTE LAS ITERS

            self.model = sm.MNLogit(y,X).fit(
                method='lbfgs',
                maxiter=max_iter,
                retall=True,
                full_output=1,
                disp=2,
            )


        elif self.model_library == 'sklearn':

            # use sklearn logistig regression
            self.model = linear_model.LogisticRegression(
                penalty='l2', C=1e5, fit_intercept=False,
                solver='lbfgs'
            )
            self.model.fit(X, y)

        else:
            print 'wrong config: {0} not in model_library'.format(
                self.model_library
            )
            sys.exit()

        elapsed_time = time.time() - start_time
        print "Optimization done in {0:.2f} seconds\n".format(elapsed_time)

    def SaveModel(self, p_save):
        'Saves fitted model for future use'

        pickle.dump(self.model, open(p_save, 'wb'))
        print 'ALR model saved at {0}'.format(p_save)

    def LoadModel(self, p_load):
        'Load fitted model'

        self.model = pickle.load(open(p_load, 'rb'))
        print 'ALR model loaded from {0}'.format(p_load)

    def Report_Fit(self, p_save):
        'Report containing model fitting info'

        # report folder
        if not op.isdir(p_save):
            os.mkdir(p_save)

        # get data 
        pval_df = self.model.pvalues.transpose()
        params_df = self.model.params.transpose()
        name_terms = pval_df.columns.tolist()
        summ = self.model.summary()

        # plot p-values
        p_plot = op.join(p_save, 'pval.png')
        Plot_PValues(pval_df.values, name_terms, p_plot)

        # plot parameters
        p_plot = op.join(p_save, 'params.png')
        Plot_Params(params_df.values, name_terms, p_plot)

        # write summary
        p_summ = op.join(p_save, 'summary.txt')
        with open(p_summ, 'w') as fW:
            fW.write(summ.as_text())

        # plot terms used for fitting
        p_terms_png = op.join(p_save, 'terms_fit')
        if not op.isdir(p_terms_png):
            os.mkdir(p_terms_png)
        self.Report_Terms_Fit()

    def Report_Terms_Fit(self):
        # TODO mostrar los terminos pc1, pc2, pc3 y season
        #terms_i = self.terms_fit
        #ss = terms_i['seasonality']
        #pc1 = terms_i['cov_1']
        #pc2 = terms_i['cov_2']
        #pc3 = terms_i['cov_3']
        #pc1_cos=terms_i['cov_1_cos']
        #pc1_sin=terms_i['cov_1_sin']
        #pc2_cos=terms_i['cov_2_cos']
        #pc2_sin=terms_i['cov_2_sin']
        #pc3_cos=terms_i['cov_3_cos']
        #pc3_sin=terms_i['cov_3_sin']

        #print pc1.shape
        #print ss.shape
        #ss1_cos = ss[:,0]
        #ss1_sin = ss[:,1]
        #ss2_cos = ss[:,2]
        #ss2_sin = ss[:,3]
        ##x = range(len(pc1))
        #x = terms_i['long_term']

        #import matplotlib.pyplot as plt
        #plt.figure(1)
        #plt.subplot(411)
        #plt.plot(
        #    x, pc1, 'k',
        #    x, pc1_cos, 'r--',
        #    x, pc1_sin, 'b--',
        #)
        #plt.title('pc1 (cos red, sin blue)')
        #plt.xlim(x[0], x[-1])
        #plt.subplot(412)
        #plt.plot(
        #    x, pc2, 'k',
        #    x, pc2_cos, 'r--',
        #    x, pc2_sin, 'b--',
        #)
        #plt.title('pc2')
        #plt.xlim(x[0], x[-1])
        #plt.subplot(413)
        #plt.plot(
        #    x, pc3, 'k',
        #    x, pc3_cos, 'r--',
        #    x, pc3_sin, 'b--',
        #)
        #plt.title('pc3')
        #plt.xlim(x[0], x[-1])
        #plt.subplot(414)
        #plt.plot(
        #    x, ss1_cos, 'r-',
        #    x, ss1_sin, 'r.',
        #    x, ss2_cos, 'b--',
        #    x, ss2_sin, 'b.',
        #)
        #plt.title('season 2pit (red) & 4pit (blue)')
        #plt.xlim(x[0], x[-1])
        #plt.show()

        pass

    def Simulate(self, num_sims, time_sim, xds_covars_sim=None):
        'Launch ARL model simulations'

        # switch library probabilities predictor function 
        if self.model_library == 'statsmodels':
            pred_prob_fun = self.model.predict
        elif self.model_library == 'sklearn':
            pred_prob_fun = self.model.predict_proba
        else:
            print 'wrong config: {0} not in model_library'.format(
                self.model_library
            )
            sys.exit()

        # get needed data
        evbmus_values = self.xds_bmus_fit.bmus.values
        time_fit = self.xds_bmus_fit.time.values
        mk_order = self.mk_order

        # print some info
        print 'ALR model fit   : {0} --- {1}'.format(
            time_fit[0], time_fit[-1])
        print 'ALR model sim   : {0} --- {1}'.format(
            time_sim[0], time_sim[-1])

        # generate time yearly fractional array
        time_yfrac = self.GetFracYears(time_sim)

        # use a d_terms_settigs copy 
        d_terms_settings_sim = self.d_terms_settings.copy()

        # start simulations
        print "\nLaunching simulations...\n"
        evbmus_sims = np.zeros((len(time_yfrac), num_sims))
        for n in range(num_sims):

            #print 'Sim. Num. {0}'.format(n+1)
            evbmus = evbmus_values[1:mk_order+1]
            for i in range(len(time_yfrac) - mk_order):

                # handle simulation covars
                if d_terms_settings_sim['covariates'][0]:

                    # simulation covariates
                    sim_covars_T = xds_covars_sim.cov_values.values

                    # normalize step covars
                    sim_covars_evbmus = sim_covars_T[i : i + mk_order +1]
                    sim_cov_norm = (
                        sim_covars_evbmus - sim_covars_T.mean(axis=0)
                    ) / sim_covars_T.std(axis=0)

                    # mount step xr.dataset for sim covariates
                    xds_cov_sim_step = xr.Dataset(
                        {
                            'cov_norm': (('time','cov_names'), sim_cov_norm),
                        },
                        coords = {
                            'cov_names': self.cov_names,
                        }
                    )

                    d_terms_settings_sim['covariates']=(True, xds_cov_sim_step)

                # generate time step ALR terms
                terms_i, _ = self.GenerateALRTerms(
                    d_terms_settings_sim,
                    np.append(evbmus[ i : i + mk_order], 0),
                    time_yfrac[i : i + mk_order + 1],
                    self.cluster_size, time2yfrac=False)


                # Event sequence simulation  (sklearn)
                X = np.concatenate(terms_i.values(),axis=1)
                prob = pred_prob_fun(X)  # statsmodels // sklearn functions
                probTrans = np.cumsum(prob[-1,:])
                evbmus = np.append(evbmus, np.where(probTrans>np.random.rand())[0][0]+1)

                # progress bar
                pb(i + 1, len(time_yfrac)-mk_order,
                   prefix = 'Sim. Num. {0:03d}'.format(n+1),
                   suffix = 'Complete', length = 50)

            evbmus_sims[:,n] = evbmus


            # Probabilities in the nsims simulations
            evbmus_prob = np.zeros((evbmus_sims.shape[0], self.cluster_size))
            for i in range(evbmus_sims.shape[0]):
                for j in range(self.cluster_size):
                    evbmus_prob[i, j] = len(np.argwhere(evbmus_sims[i,:]==j+1))/float(num_sims)

        evbmus_probcum = np.cumsum(evbmus_prob, axis=1)

        # return ALR simulation data in a xr.Dataset
        return xr.Dataset(
            {
                'evbmus_sims': (('time', 'n_sim'), evbmus_sims),
                'evbmus_probcum': (('time', 'n_cluster'), evbmus_probcum),
            },

            coords = {
                'time' : [np.datetime64(d) for d in time_sim],
            },
        )

    def Report_Sim(self, xds_ALR_sim, p_save, xds_cov_sim=None):
        '''
        Report containing model fitting report.
        Compares fitting data to simulated output (xds_ALR_sim)
        '''

        # report folder and files
        if not op.isdir(p_save):
            os.mkdir(p_save)

        # get data 
        # TODO: LAS CONVERSIONES DE DATETIME SON MUY LENTAS
        cluster_size = self.cluster_size
        bmus_values_sim = xds_ALR_sim.evbmus_sims.values
        bmus_dates_sim = [x2d(t) for t in xds_ALR_sim.time]
        bmus_values_hist = np.reshape(self.xds_bmus_fit.bmus.values,[-1,1])
        bmus_dates_hist = [x2d(t) for t in self.xds_bmus_fit.time]
        num_sims = bmus_values_sim.shape[1]

        # Plot Perpetual Year - bmus wt
        p_rep_PP = op.join(p_save, 'perp_year.png')
        Plot_Compare_PerpYear(
            cluster_size,
            bmus_values_sim, bmus_dates_sim,
            bmus_values_hist, bmus_dates_hist,
            n_sim = num_sims, p_export=p_rep_PP
        )

        if self.d_terms_settings['covariates'][0]:

            # TODO comprobar esto de tiempos duplicados
            time_hist_covars = bmus_dates_hist
            time_sim_covars = bmus_dates_sim

            # covars fit
            xds_cov_fit = self.d_terms_settings['covariates'][1]
            cov_names = xds_cov_fit.cov_names.values
            cov_fit_values = xds_cov_fit.cov_values.values

            # covars sim
            cov_sim_values = xds_cov_sim.cov_values.values

            # TODO: QUITAR
            cov_names = ['PC1', 'PC2', 'PC3']

            for ic, cn in enumerate(cov_names):

                # get covariate data
                cf_val = cov_fit_values[:,ic]
                cs_val = cov_sim_values[:,ic]

                # plot covariate - bmus wt
                # TODO: NO ESTOY SEGURO QUE LO ESTE HACIENDO BIEN
                # CUANDO ESTE MAS TRANQUILO MIRAR LO DE LOS YEARS DE LAS DOS
                # FECHAS DENTRO
                p_rep_cn = op.join(p_save, '{0}_comp.png'.format(cn))
                Plot_Compare_Covariate(
                    cluster_size,
                    bmus_values_sim, bmus_dates_sim,
                    bmus_values_hist, bmus_dates_hist,
                    cs_val, time_sim_covars,
                    cf_val, time_hist_covars,
                    cn,
                    n_sim = num_sims, p_export = p_rep_cn
                )

