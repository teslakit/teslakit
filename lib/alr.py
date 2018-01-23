#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
np.set_printoptions(threshold=np.nan)
from sklearn import linear_model
from collections import OrderedDict

def Generate_ALRTerms(bmus, clust_size, mk_order, t, cov, covT):
    '''
    Creates terms for ALR model. Returns OrderedDict with ALR terms.

    bmus - cluster evolution data
    clust_size - number of clusters
    mk_order - markov chain order
    t - time series
    cov, covT - covariables
    '''

    # return an Ordered dictonary
    terms = OrderedDict()

    # constant term
    terms['constant'] = np.ones((bmus.size,1))

    # time and seasonality term
    if t:
        #T = np.zeros((bmus.size,1))
        #T[:,0] = t
        terms['seasonality'] = np.column_stack([np.cos(2*np.pi*t), np.sin(2*np.pi*t)])

    # Covariables term (normalization)
    if cov and covT:
        alrt_covar = (cov - covT.mean(axis=0)) / covT.std(axis=0)
        for i in range(alrt_covar.shape[1]):
            terms['cov_{0}'.format(i+1)] = np.transpose(np.asmatrix(covN[:,i]))

    # dummi for markov chain
    def dummi(csize):
        D = np.ones((csize-1, csize)) * -1
        for i in range(csize-1):
            D[i, csize-1-i] = csize-i-1
            D[i, csize-1+1-i:] = 0
        return D
    dum = dummi(clust_size)

    # solve markov order N
    for i in range(mk_order):
        Z = np.zeros((bmus.size, clust_size-1))
        for indz in range(bmus.size-i-1):
            Z[indz+i+1,0:] = np.squeeze(dum[0:,bmus[indz]-1])
        terms['markov_{0}'.format(i+1)] = Z

    return terms

def AutoRegLogisticReg(evbmus, cluster_size, num_sims, sim_start_y, sim_end_y):
    '''
    TODO: definir lo que hace

    evbmus - KMA classification bmus
    cluster_size - number of states
    num_sims - number of simulations
    sim_start_y - simulation start year
    sim_end_y - simulation end year
    '''

    mk_order = 1  #TODO  decidir si va fuera o dentro

    # TODO: INTRODUCIR OPCION FITTING PERIOD? O USAR TODOS LOS DATOS EN BMUS

    # initialize model fitting
    terms = Generate_ALRTerms(
        evbmus, cluster_size, mk_order, None, None, None)

    # start model fitting
    tmodl = np.concatenate(terms.values(), axis=1)
    alr = linear_model.LogisticRegression(penalty='l2',C=1e5,fit_intercept=False)
    alr.fit(tmodl,evbmus)
    predprob = alr.predict_proba(tmodl)


    # start model simulations
    list_sim_years = range(sim_start_y, sim_end_y)

    evbmusd_sims = np.zeros((len(list_sim_years), num_sims))
    for n in range(num_sims):
        print 'simulation num. {0}'.format(n+1)
        evbmusd = evbmus[:2]  # simulation bmus start from KMA

        for i in range(len(list_sim_years) -mk_order-1):

            terms = Generate_ALRTerms(
                np.append(evbmusd[i : i + mk_order], 0),
                cluster_size, mk_order, None, None, None)

            # Event sequence simulation   
            prob = alr.predict_proba(np.concatenate(terms.values(),axis=1))
            probTrans = np.cumsum(prob[-1,:])
            evbmusd = np.append(evbmusd, np.where(probTrans>np.random.rand())[0][0]+1)

        evbmusd_sims[:,n] = evbmusd

    # Probabilities in the nsims simulations
    evbmus_prob = np.zeros((evbmusd_sims.shape[0], cluster_size))
    for i in range(evbmusd_sims.shape[0]):
        for j in range(cluster_size):
            evbmus_prob[i, j] = len(np.argwhere(evbmusd_sims[i,:]==j+1))/float(num_sims)

    evbmus_probcum = np.cumsum(evbmus_prob, axis=1)

    return evbmusd_sims

