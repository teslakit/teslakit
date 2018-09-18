#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__),'..'))

# python libs
import numpy as np
import xarray as xr
from datetime import date, timedelta, datetime

# tk libs
from lib.objs.tkpaths import PathControl
from lib.mjo import GetMJOCategories
from lib.plotting.MJO import Plot_MJOphases, Plot_MJOCategories
from lib.objs.alr_wrapper import ALR_WRP


# --------------------------------------
# data storage and path control
pc = PathControl()
pc.SetSite('test_site')


# --------------------------------------
# Load MJO data (previously downloaded)
xds_mjo_hist = xr.open_dataset(pc.DB.mjo.hist)


# --------------------------------------
# Calculate MJO categories (25 used) 
rmm1 = xds_mjo_hist['rmm1']
rmm2 = xds_mjo_hist['rmm2']
phase = xds_mjo_hist['phase']

categ, d_rmm_categ = GetMJOCategories(rmm1, rmm2, phase)
xds_mjo_hist['categ'] = (('time',), categ)


# plot MJO phases
p_export = op.join(pc.DB.tmp.export_figs, 'mjo_phases')  # if only show: None
Plot_MJOphases(rmm1, rmm2, phase, p_export)

# plot MJO categories
p_export = op.join(pc.DB.tmp.export_figs, 'mjo_categ')  # if only show: None
Plot_MJOCategories(rmm1, rmm2, categ, p_export)


# --------------------------------------
# Autoregressive Logistic Regression

# MJO historical data for fitting
num_categs  = 25
xds_bmus_fit = xds_mjo_hist.categ


# Autoregressive logistic enveloper
ALRW = ALR_WRP(xds_bmus_fit, num_categs)

# ALR terms
d_terms_settings = {
    'mk_order'  : 3,
    'constant' : True,
    'seasonality': (True, [2,4,8]),
}

ALRW.SetFittingTerms(d_terms_settings)

# ALR model fitting
ALRW.FitModel(max_iter=10000)

# ALR model simulations 
sim_num = 1  # only one simulation for mjo daily
sim_years = 500

# simulation dates
d1 = date(1700,6,1)
d2 = date(d1.year+sim_years, d1.month, d1.day)
dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

# launch simulation
xds_alr = ALRW.Simulate(sim_num, dates_sim)
evbmus_sim = xds_alr.evbmus_sims.values

# parse to 1D array
evbmus_sim = np.squeeze(evbmus_sim)

# Generate mjo_sim list using random mjo from each category
# TODO: MUY LENTO, ACELERAR
print 'parte avisada como muy lenta'
mjo_sim = np.empty((len(evbmus_sim),2)) * np.nan
for c, m in enumerate(evbmus_sim):
    options = d_rmm_categ['cat_{0}'.format(int(m))]
    r = np.random.randint(options.shape[0])
    mjo_sim[c,:] = options[r,:]

# TODO COMO OBTENGO MJO SIMULATED PHASE? RMM1? RMM2?


# TODO: GUARDAR MJO SIMULADO EN XARRAY 
xds_MJO_sim = xr.Dataset(
    {
        'mjo'   :(('time',), mjo_sim),
        #'phase' :(('time',), phase_sim),
        #'rmm1'  :(('time',), rmm1_sim),
        #'rmm2'  :(('time',), rmm2_sim),
    },
    {'time' : dates_sim}
)

