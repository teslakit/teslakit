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
pc.SetSite('KWAJALEIN')


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
p_export = op.join(pc.site.export_figs, 'MJO', 'mjo_phases')  # if only show: None
Plot_MJOphases(rmm1, rmm2, phase, p_export)

# plot MJO categories
p_export = op.join(pc.site.export_figs, 'MJO', 'mjo_categ')  # if only show: None
Plot_MJOCategories(rmm1, rmm2, categ, p_export)


# --------------------------------------
# Autoregressive Logistic Regression - fit model

# MJO historical data for fitting
num_categs  = 25

xds_bmus_fit = xr.Dataset(
    {
        'bmus'  :(('time',), xds_mjo_hist.categ.values[:]),
    },
    {'time' : xds_mjo_hist.time}
)

# ALR terms
d_terms_settings = {
    'mk_order'  : 3,
    'constant' : True,
    'seasonality': (True, [2,4,8]),
}

# ALR wrapper
ALRW = ALR_WRP(pc.site.mjo.alrw)
ALRW.SetFitData(num_categs, xds_bmus_fit, d_terms_settings)

# ALR model fitting
ALRW.FitModel(max_iter=10000)


# --------------------------------------
# Autoregressive Logistic Regression - simulate 

# simulation dates
d1 = date(1700,6,1)
d2 = date(2200,6,1)
dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

# launch simulation
sim_num = 1  # only one simulation for mjo daily
xds_alr = ALRW.Simulate(sim_num, dates_sim)
evbmus_sim = np.squeeze(xds_alr.evbmus_sims.values[:])


# Generate mjo_sim, rmm12_sim, phase_sim using random mjo value from each category
rmm12_sim = np.empty((len(evbmus_sim),2)) * np.nan
for c, m in enumerate(evbmus_sim):
    # rmm1, rmm2
    options = d_rmm_categ['cat_{0}'.format(int(m))]
    r = np.random.randint(options.shape[0])
    rmm12_sim[c,:] = options[r,:]


# store simulated mjo
# TODO: como simulo la phase y el mjo? igual que las componentes rmm12?
xds_MJO_sim = xr.Dataset(
    {
        #'mjo'   :(('time',), mjo_sim),
        #'phase' :(('time',), phase_sim),
        'rmm1'  :(('time',), rmm12_sim[:,0]),
        'rmm2'  :(('time',), rmm12_sim[:,1]),
    },
    {'time' : [np.datetime64(d) for d in dates_sim]}
)
print xds_MJO_sim
xds_MJO_sim.to_netcdf(pc.site.mjo.sim, 'w')

