#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
np.set_printoptions(threshold=np.nan)
from sklearn import linear_model
from collections import OrderedDict
from datetime import datetime, date
import xarray as xr

# TODO: ACLARAR GESTION DEL DATO TIME

def GetNumYears(time_data):
    'Returns time in custom year decimal format'

    d0 = date(
        time_data[0].dt.year,
        time_data[0].dt.month,
        time_data[0].dt.day)
    d1 = date(
        time_data[-1].dt.year,
        time_data[-1].dt.month,
        time_data[-1].dt.day)

    delta = d1- d0
    return  np.array(range(delta.days+1))/365.25

#def Generate_ALRTerms(bmus, clust_size, mk_order, time_data, season_data, cov, covT):
def Generate_ALRTerms(bmus, clust_size, d_alrterms):
    '''
    Creates terms for ALR model. Returns OrderedDict with ALR terms.

    bmus - cluster evolution data
    clust_size - number of clusters
    d_alrterms - ALR terms options
    '''

    # return an Ordered dictonary
    terms = OrderedDict()

    # constant term
    if d_alrterms['constant_term'][0]:
        terms['constant'] = np.ones((bmus.size,1))

    # Time term
    # TODO: creo que este termino no funciona bien (14000,1) --> (14000,)
    # time array (use custom time array with year decimals)
    if d_alrterms['time_term'][0]:
        terms['time'] = GetNumYears(d_alrterms['time_term'][1])

    # Seasonality term 
    # time array (use custom time array with year decimals)
    if d_alrterms['seasonality_term'][0]:
        season_time = GetNumYears(d_alrterms['seasonality_term'][1])
        season_amps = d_alrterms['seasonality_term'][2]

        temp_seas = np.zeros((len(season_time), 2*len(season_amps)))
        c = 0
        for amp in season_amps:
            temp_seas [:,c]   = np.cos(amp*np.pi*season_time)
            temp_seas [:,c+1] = np.sin(amp*np.pi*season_time)
            c+=2
        terms['seasonality'] = temp_seas

    # TODO: desarrollar covariables term
    # Covariables term (normalization)
    #if cov and covT:
    #    alrt_covar = (cov - covT.mean(axis=0)) / covT.std(axis=0)
    #    for i in range(alrt_covar.shape[1]):
    #        terms['cov_{0}'.format(i+1)] = np.transpose(np.asmatrix(covN[:,i]))

    # markov term
    if d_alrterms['mk_order'][0]:
        # dummi for markov chain
        def dummi(csize):
            D = np.ones((csize-1, csize)) * -1
            for i in range(csize-1):
                D[i, csize-1-i] = csize-i-1
                D[i, csize-1+1-i:] = 0
            return D
        dum = dummi(clust_size)

        # solve markov order N
        mk_order = d_alrterms['mk_order'][1]
        for i in range(mk_order):
            Z = np.zeros((bmus.size, clust_size-1))
            for indz in range(bmus.size-i-1):
                Z[indz+i+1,0:] = np.squeeze(dum[0:,bmus[indz]-1])
            terms['markov_{0}'.format(i+1)] = Z

    return terms

def AutoRegLogisticReg(evbmus, cluster_size, num_sims, sim_start_y, sim_end_y,
                       d_alrterms = {}):
    '''
    TODO: definir

    evbmus          - KMA classification bmus
    cluster_size    - number of states
    num_sims        - number of simulations
    sim_start_y     - simulation start year
    sim_end_y       - simulation end year
    d_alrterms      - ALR terms parameters
    '''

    # TODO: GENERALIZAR EL VECTOR TIEMPO PARA SER ANUAL, DAILY, O FUNCION DEL
    # INPUT DE USUARIO
    # TODO: PUEDE QUE HAYA QUE DAR: FECHA INICIAL, FECHA FINAL, RESOLUCION
    # DAILY / YEARLY

    # default ALR terms
    d_alrterms_default = {
        'mk_order'  : (True, 1),        # markov tree order (bool, mk_order)
        'constant_term' : (True,),      # constant term     (bool,)
        'time_term' : (False, ),        # time term         (bool, time_data)
        'seasonality_term': (False,0,), # seasonality term  (bool, time_data, amplitudes)
    }

    # join user and default input
    for k in d_alrterms_default.keys():
        if k not in d_alrterms:
            d_alrterms[k] = d_alrterms_default[k]


    # initialize model fitting
    terms = Generate_ALRTerms(evbmus, cluster_size, d_alrterms)

    # start model fitting
    tmodl = np.concatenate(terms.values(), axis=1)
    alr = linear_model.LogisticRegression(penalty='l2',C=1e5,fit_intercept=False)
    alr.fit(tmodl,evbmus)
    predprob = alr.predict_proba(tmodl)

    # get some data before simulations
    if d_alrterms['time_term'][0]:
        time_term_tarray = d_alrterms['time_term'][1]

    if d_alrterms['seasonality_term'][0]:
        season_time_array = d_alrterms['seasonality_term'][1]
        season_amps = d_alrterms['seasonality_term'][2]

    # start model simulations
    mk_order = d_alrterms['mk_order'][1]
    list_sim_years = range(sim_start_y, sim_end_y)

    evbmusd_sims = np.zeros((len(list_sim_years), num_sims))
    print "Computing {0} simulations...".format(num_sims)
    for n in range(num_sims):
        print 'simulation num. {0}'.format(n+1)
        evbmusd = evbmus[:mk_order]  # simulation bmus start from KMA
        # TODO: ? usar evbmus[:mk_order] (los primeros) o [-mk_order:] (los ultimos)

        for i in range(len(list_sim_years) - mk_order):

            # fix ALR terms for simulation (time related terms)
            d_sim_alrterms = d_alrterms

            if d_sim_alrterms['time_term'][0]:
                d_sim_alrterms['time_term'] = (
                    True,
                    time_term_tarray[i : i + mk_order + 1])

            if d_sim_alrterms['seasonality_term'][0]:
                d_sim_alrterms['seasonality_term'] = (
                    True,
                    season_time_array[i : i + mk_order + 1],
                    season_amps
                )

            terms = Generate_ALRTerms(
                np.append(evbmus[i : i + mk_order], 0),
                cluster_size, d_sim_alrterms)


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

    return evbmusd_sims, evbmus_probcum

