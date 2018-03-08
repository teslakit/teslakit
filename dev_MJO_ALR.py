#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op
import xarray as xr

from lib.mjo import GetMJOCategories, DownloadMJO
from lib.custom_plot import Plot_MJOphases, Plot_MJOCategories
from lib.objs.alr_enveloper import ALR_ENV

# data storage
p_data = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/'

p_mjo_hist = op.join(p_data, 'historical', 'MJO_hist.nc')


# ---------------------------------
# Download mjo and mount xarray.dataset
#y1 = '1979-01-01'
#ds_mjo_hist = DownloadMJO(p_mjo_hist, init_year=y1)

# Load MJO data (previously downloaded)
ds_mjo_hist = xr.open_dataset(p_mjo_hist)


# ---------------------------------
# Calculate MJO categories (25 used) 
rmm1 = ds_mjo_hist['rmm1']
rmm2 = ds_mjo_hist['rmm2']
phase = ds_mjo_hist['phase']

categ, d_rmm_categ = GetMJOCategories(rmm1, rmm2, phase)


# plot MJO data
Plot_MJOphases(rmm1, rmm2, phase)

# plot MJO categories
Plot_MJOCategories(rmm1, rmm2, categ)



# ---------------------------------
# Autoregressive Logistic Regression
# TODO: adaptar a los cambios de ALR_ENVELOPER

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
    'mk_order'  : 0,  # markov 0 for MJO
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


# TODO: AHORA TENGO QUE PASAR EVBMUSSIM A UNA SERIE DE DATOS RMM1.RMM2 
# GENERADA "PSEUDOALEATORIAMENTE" "CAYENDO EN LAS CATEGORIAS Y COGIENDO UNO AL
# AZAR" DWT_PREPARE_MJO_AWT_2P
