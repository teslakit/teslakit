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
# get test data from base .mat files 

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
i0 = dates_MJO.index(date(1979,1,1))
rmm1 = rmm1[i0:]
rmm2 = rmm2[i0:]
dates_MJO = dates_MJO[i0:]


# AWT: PCs
p_mat = op.join(p_data, 'PCs_for_AWT.mat')
d_mat = rmat(p_mat)['AWT']
PCs = d_mat['PCs']
dates_AWT = [date(r[0],r[1],r[2]) for r in d_mat['Dates']]
# parse annual data to daily data (using pandas)
temp = pd.DataFrame(PCs[:,0:3], columns=['PC1','PC2','PC3'], index=dates_AWT)
PCs_d = temp.reindex(pd.date_range(start=temp.index[0],end=temp.index[-1],freq='D'),method='pad')
dates_AWT_d = temp.index


# make covar data share the same dates
# TODO: LAS COVAR DEL MJO USAN LA FECHA INICIA AWT. COMENTAR EQUIPO
date_ini = max(dates_MJO[0], dates_AWT[0], dates_KMA[0])
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
cov_1 = cov_PCs.PC1.reshape(-1,1)
cov_2 = cov_PCs.PC2.reshape(-1,1)
cov_3 = cov_PCs.PC3.reshape(-1,1)

# MJO covars
i0 = next(x[0] for x in enumerate(dates_MJO) if x[1] > dates_covar[0])-1
i1 = next(x[0] for x in enumerate(reversed(dates_MJO)) if x[1] < dates_covar[-1])
i1 = len(dates_MJO)-i1+1
cov_4 = np.array(rmm1[i0:i1]).reshape(-1,1)
cov_5 = np.array(rmm2[i0:i1]).reshape(-1,1)

# join covars
cov = np.hstack((cov_1, cov_2, cov_3, cov_4, cov_5))
# TODO totcov, que es? 



# TODO: SEGUIR AQUI, TENEMOS COVAR, HAY QUE ANADIRLO A LA LIBRERIA ALR
# EL RESTO DEL CODIGO ES PLACEHOLDER COPIADO DE TESTMJOALR



# -----------------------------------
# Autoregressive Logistic Regression

# Load a MJO data from netcdf
p_mjo_cut = op.join(p_data, 'MJO_categ.nc')
ds_mjo_cut = xr.open_dataset(p_mjo_cut)

bmus = ds_mjo_cut['categ'].values
t_data = ds_mjo_cut['time']
num_categs  = 25

# Autoregressive logistic enveloper
ALRE = ALR_ENV(bmus, t_data, num_categs)

# ALR terms
d_terms_settings = {
    'mk_order'  : 3,
    'constant' : True,
    'long_term' : False,
    'seasonality': (True, [2,4,8]),
}

ALRE.SetFittingTerms(d_terms_settings)

# ALR model fitting
ALRE.FitModel()

# ALR model simulations 
sim_num = 1
sim_start = 1900
sim_end = 2602
sim_freq = '1d'

evbmus_sim, evbmus_probcum = ALRE.Simulate(sim_num, sim_start, sim_end,
                                           sim_freq)

print evbmus_sim
print evbmus_probcum

