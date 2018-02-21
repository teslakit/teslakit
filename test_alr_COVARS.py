#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op
import xarray as xr
import pandas as pd
import numpy as np
from lib.objs.alr_enveloper import ALR_ENV
from lib.io.matlab import ReadMatfile as rmat
from lib.custom_dateutils import DateConverter_Mat2Py
from datetime import date, timedelta, datetime

# data storage
p_data = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/tests_ALR/'


# ---------------------------------
# get test data from base .mat demo files 

# KMA: bmus
p_mat = op.join(p_data, 'KMA_daily_42.mat')
d_mat = rmat(p_mat)['KMA']
bmus = d_mat['bmus']
dates_KMA = [date(r[0],r[1],r[2]) for r in d_mat['Dates']]


# MJO: rmm1, rmm2
p_mat = op.join(p_data, 'MJO.mat')
d_mat = rmat(p_mat)
rmm1 = d_mat['rmm1']
rmm2 = d_mat['rmm2']
dates_MJO = [date(r[0],r[1],r[2]) for r in d_mat['Dates']]
# remove MJO nans (before 1979)
rmm = np.vstack((rmm1, rmm2)).transpose()
temp = pd.DataFrame(rmm, columns=['rmm1','rmm2'], index=dates_MJO)
MJO_d = temp.reindex(pd.date_range(start=datetime(1979,01,01),end=temp.index[-1],freq='D'),method='pad') # there are nans in 1978
dates_MJO = [x.date() for x in MJO_d.index] # update MJO dates


# AWT: PCs
p_mat = op.join(p_data, 'PCs_for_AWT_mes10.mat')
d_mat = rmat(p_mat)['AWT']
PCs = d_mat['PCs']
dates_AWT = [date(r[0],r[1],r[2]) for r in d_mat['Dates']]
# parse annual data to daily data (using pandas)
temp = pd.DataFrame(PCs[:,0:3], columns=['PC1','PC2','PC3'], index=dates_AWT)
PCs_d = temp.reindex(pd.date_range(start=temp.index[0],end=temp.index[-1],freq='D'),method='pad')
dates_AWT_d = [x.date() for x in PCs_d.index]


# make covar data share the same dates
date_ini = max(dates_MJO[0], dates_AWT[0])
date_end = min(dates_MJO[-1], dates_AWT[-1])
dates_covar = [date_ini + timedelta(days=i) for i in
               range((date_end-date_ini).days+1)]

print 'KMA dates:   {0} --- {1}'.format(dates_KMA[0], dates_KMA[-1])
print ''
print 'COVARIATES:'
print 'MJO dates:   {0} --- {1}'.format(dates_MJO[0], dates_MJO[-1])
print 'AWT dates:   {0} --- {1}'.format(dates_AWT[0], dates_AWT[-1])
print 'covar dates: {0} --- {1}'.format(dates_covar[0], dates_covar[-1])


# ---------------------------------
# Mount covariates

# PCs covar 
cov_PCs = PCs_d.loc[dates_covar[0]:dates_covar[-1]]
cov_1 = cov_PCs.PC1.values.reshape(-1,1)
cov_2 = cov_PCs.PC2.values.reshape(-1,1)
cov_3 = cov_PCs.PC3.values.reshape(-1,1)

# MJO covars
cov_MJO = MJO_d.loc[dates_covar[0]:dates_covar[-1]]
cov_4 = cov_MJO.rmm1.values.reshape(-1,1)
cov_5 = cov_MJO.rmm2.values.reshape(-1,1)

# join covars and norm.
cov_T = np.hstack((cov_1, cov_2, cov_3, cov_4, cov_5))

# KMA related covars starting at KMA period 
i0 = dates_covar.index(dates_KMA[0])
cov_KMA = cov_T[i0:,:]

# normalize
cov_norm = (cov_KMA - cov_T.mean(axis=0)) / cov_T.std(axis=0)


# -----------------------------------
# Autoregressive Logistic Regression

# use bmus inside covariate time frame
i0 = dates_KMA.index(max(dates_covar[0], dates_KMA[0]))
i1 = dates_KMA.index(min(dates_covar[-1], dates_KMA[-1]))+1
bmus = bmus[i0 : i1]
t_data = dates_KMA[i0 : i1]
num_clusters  = 42


# Autoregressive logistic enveloper
ALRE = ALR_ENV(bmus, t_data, num_clusters)

# ALR terms
d_terms_settings = {
    'mk_order'  : 3,
    'constant' : True,
    'long_term' : False,
    'seasonality': (True, [2]),
    'covariates': (True, cov_norm),
}

ALRE.SetFittingTerms(d_terms_settings)

# ALR model fitting
ALRE.FitModel()



# TODO: simular 3 years de test

# ---------------------------------
# ALR model simulations 
sim_num = 10
sim_start = 1700
sim_end = 1703
sim_freq = '1d'


# ---------------------------------
# get covariates data for simulation

# AWT: PCs
p_mat = op.join(p_data, 'AWT_PCs_500_part1.mat')
d_mat = rmat(p_mat)['AWT']
PCs = d_mat['PCs']
dates_AWT = [date(r[0],r[1],r[2]) for r in d_mat['Dates']]
temp = pd.DataFrame(PCs[:,0:3], columns=['PC1','PC2','PC3'], index=dates_AWT)
PCs_d_sim = temp.reindex(pd.date_range(start=temp.index[0],end=temp.index[-1],freq='D'),method='pad')
dates_AWT_d_sim = [x.date() for x in PCs_d_sim.index]

# MJO: rmm1, rmm2
p_mat = op.join(p_data, 'MJO_500_part1.mat')
d_mat = rmat(p_mat)
rmm1 = d_mat['rmm1']
rmm2 = d_mat['rmm2']
dates_MJO = [date(r[0],r[1],r[2]) for r in d_mat['Dates']]
rmm = np.vstack((rmm1, rmm2)).transpose()
temp = pd.DataFrame(rmm, columns=['rmm1','rmm2'], index=dates_MJO)
MJO_d_sim = temp.reindex(pd.date_range(start=temp.index[0],end=temp.index[-1],freq='D'),method='pad') # there are nans in 1978

# make covar data share the same dates
date_ini = max(dates_MJO[0], dates_AWT[0])
date_end = min(dates_MJO[-1], dates_AWT[-1])
dates_covar_sim = [date_ini + timedelta(days=i) for i in
                   range((date_end-date_ini).days+1)]


# ---------------------------------
# Mount simulation covariates

# PCs covar 
cov_PCs = PCs_d_sim.loc[dates_covar_sim[0]:dates_covar_sim[-1]]
cov_1 = cov_PCs.PC1.values.reshape(-1,1)
cov_2 = cov_PCs.PC2.values.reshape(-1,1)
cov_3 = cov_PCs.PC3.values.reshape(-1,1)

# MJO covars
cov_MJO = MJO_d_sim.loc[dates_covar_sim[0]:dates_covar_sim[-1]]
cov_4 = cov_MJO.rmm1.values.reshape(-1,1)
cov_5 = cov_MJO.rmm2.values.reshape(-1,1)

# join covars and norm.
cov_T_sim = np.hstack((cov_1, cov_2, cov_3, cov_4, cov_5))


# launch simulation
evbmus_sim, evbmus_probcum, dump = ALRE.Simulate(
    sim_num, sim_start, sim_end, sim_freq, cov_T_sim)

print evbmus_sim
print evbmus_probcum

