#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op

# pip
import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
plt.rcParams['figure.figsize'] = [18, 8]
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# DEV: override installed teslakit
#import sys
#sys.path.insert(0,'../../')

# teslakit 
from teslakit.project_site import Site
from teslakit.io.matlab import ReadGowMat, ReadCoastMat, ReadEstelaMat
from teslakit.estela import spatial_gradient, mask_from_poly, Predictor
from teslakit.plotting.estela import Plot_ESTELA_Globe


# --------------------------------------
# Site paths and parameters
data_folder = r'/Users/nico/Projects/TESLA-kit/TeslaKit/data'
site = Site(data_folder, 'KWAJALEIN_TEST')

DB = site.pc.DB                               # common database
ST = site.pc.site                             # site database
PR = site.params                              # site parameters

# input files
p_est_coastmat = ST.ESTELA.coastmat           # estela coast (.mat)
p_est_datamat = ST.ESTELA.estelamat           # estela data (.mat)
p_gow_mat = ST.ESTELA.gowpoint                # gow point (.mat)
p_wvs_parts_p1 = ST.WAVES.partitions_p1       # waves partitions data (GOW)
p_slp = ST.ESTELA.slp                         # site slp data (.nc)
p_hist_r2_params = ST.TCs.hist_r2_params      # WMO historical TCs parameters at r2


# output files
p_est_pred = ST.ESTELA.pred_slp               # estela slp predictor
p_mutau_wt = ST.ESTELA.hydrog_mutau           # intradaily WTs mu,tau data folder


# parameters for KMA_REGRESSION_GUIDED
kma_date_ini = PR.ESTELA_KMA_RG.date_ini
kma_date_end = PR.ESTELA_KMA_RG.date_end
num_clusters = int(PR.ESTELA_KMA_RG.num_clusters)
kmarg_alpha = float(PR.ESTELA_KMA_RG.alpha)
pnt_lon = float(PR.WAVES.point_longitude)
pnt_lat = float(PR.WAVES.point_latitude)


## --------------------------------------
## Extract predictor data (SLP) from CFSR and save to .nc file
#
## SLP extraction coordinates 
#lat1 = float(PR.SLP.lat1)
#lat2 = float(PR.SLP.lat2)
#lon1 = float(PR.SLP.lon1)
#lon2 = float(PR.SLP.lon2)
#resample = int(PR.SLP.resample)  # 2º resolution
#
## CFSR prmsl database
#p_DB_cfs_prmsl = DB.CFS.cfs_prmsl
#
#xds_SLP_site = ReadSLP(
#    p_DB_cfs_prmsl,
#    lat1, lat2, lon1, lon2, resample,
#    p_save=p_site_SLP)


# --------------------------------------
# load sea polygons for mask generation
ls_sea_poly = ReadCoastMat(p_est_coastmat)

# load estela data
xds_est = ReadEstelaMat(p_est_datamat)
estela_D = xds_est.D_y1993to2012


# plot estela days and wave point
Plot_ESTELA_Globe(pnt_lon, pnt_lat, estela_D)


# --------------------------------------
# load waves data from .mat gow file
xds_WAVES = ReadGowMat(p_gow_mat)
print('\nResampling waves data to daily mean...')

# calculate Fe (from GOW waves data)
hs = xds_WAVES.hs
tm = xds_WAVES.t02
Fe = np.multiply(hs**2,tm)**(1.0/3)
xds_WAVES.update({
    'Fe':(('time',), Fe)
})

# select time window and do data daily mean
xds_WAVES = xds_WAVES.sel(
    time = slice(kma_date_ini, kma_date_end)
).resample(time='1D').mean()
print(xds_WAVES)


# --------------------------------------
# load SLP (use xarray saved predictor data)
xds_SLP_site = xr.open_dataset(p_slp)

# select time window and do data daily mean
print('\nResampling SLP data to daily mean...')
xds_SLP_day = xds_SLP_site.sel(
    time = slice(kma_date_ini, kma_date_end)
).resample(time='1D').mean()
print('SLP: ',xds_SLP_day.time.values[0],' - ',xds_SLP_day.time.values[-1])

# calculate daily gradients
print('\nCalculating SLP spatial gradient...')
xds_SLP_day = spatial_gradient(xds_SLP_day, 'SLP')

# generate land mask with land polygons
print('\nReading land mask and ESTELA data for SLP...')
xds_SLP_day = mask_from_poly(
    xds_SLP_day, ls_sea_poly, 'mask_land')

# resample estela to site mesh
xds_est_site = xds_est.sel(
    longitude = xds_SLP_site.longitude,
    latitude = xds_SLP_site.latitude,
)
estela_D = xds_est_site.D_y1993to2012

# generate SLP mask using estela
mask_est = np.zeros(estela_D.shape)
mask_est[np.where(estela_D<1000000000)]=1

xds_SLP_day.update({
    'mask_estela':(('latitude','longitude'), mask_est)
})
print(xds_SLP_day)


# plot SLP predictor and SLP_gradient
tp = 150
xds_SLP_day.SLP.isel(time=tp).where(xds_SLP_day.mask_estela==1).plot()
plt.show()

xds_SLP_day.SLP_gradient.isel(time=tp).where(xds_SLP_day.mask_estela==1).plot()
plt.show()


# --------------------------------------
# Use a tesla-kit predictor object
pred = Predictor(p_est_pred)
pred.data = xds_SLP_day

# Calculate PCA (dynamic estela predictor)
print('\nPrincipal Component Analysis (ESTELA predictor)...')
pred.Calc_PCA_EstelaPred('SLP', estela_D)

# save predictor data
pred.Save()


# plot PCA EOFs
n_EOFs = 3
pred.Plot_EOFs_EstelaPred(n_EOFs, show=True)


# Calculate KMA (regression guided with WAVES data)
print('\nKMA Classification (regression guided: waves)...')

# TODO: encontrar alpha optimo?
pred.Calc_KMA_regressionguided(
    num_clusters,
    xds_WAVES, ['hs','t02','Fe'],
    kmarg_alpha)

# save predictor data
pred.Save()

# plot KMA clusters
pred.Plot_KMArg_clusters_datamean('SLP', show=True, mask_name='mask_estela')


# --------------------------------------
# load historical storms-parameters inside r2, and modify predictor KMA
xds_TCs_r2_params = xr.open_dataset(p_hist_r2_params)

storm_dates = xds_TCs_r2_params.dmin_date.values[:]
storm_categs = xds_TCs_r2_params.category.values[:]

# modify predictor KMA with circle storms data
print('\nAdding Historical TCs to SLP_PREDICTOR KMA_RG bmus...')
pred.Mod_KMA_AddStorms(storm_dates, storm_categs)


# --------------------------------------
# Calculate intradaily MU TAU hydrographs

xds_WAVES_p1 = ReadGowMat(p_wvs_parts_p1)
l_xds_MUTAU = pred.Calc_MU_TAU_Hydrographs(xds_WAVES_p1)

# TODO: plot report hydrographs

# store hydrographs MU TAU
if not op.isdir(p_mutau_wt): os.makedirs(p_mutau_wt)
for x in l_xds_MUTAU:
    n_store = 'MUTAU_WT{0:02}.nc'.format(x.WT)
    x.to_netcdf(op.join(p_mutau_wt, n_store), 'w')

print('MU TAU hydrographs stored.')

