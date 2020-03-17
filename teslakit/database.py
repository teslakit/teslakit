#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op
import configparser
from shutil import copyfile
import pickle
import json
from datetime import timedelta

# pip
from prettytable import PrettyTable
import numpy as np
import pandas as pd
import xarray as xr

# teslakit
from .__init__ import __version__, __author__
from .io.getinfo import description
from .io.aux_nc import StoreBugXdset
from .io.matlab import ReadTCsSimulations, ReadMatfile, ReadNakajoMats, \
ReadGowMat, ReadCoastMat, ReadEstelaMat

from .util.time_operations import xds_reindex_daily, xds_reindex_monthly, \
xds_limit_dates, xds_common_dates_daily, fast_reindex_hourly, \
generate_datetimes


# TODO: change all historical data to standarized .nc files
# TODO use this xds['time'] = [d2d(x) for x in xds.time.values[:]]


def clean_files(l_files):
    'remove files at list'
    for f in l_files:
        if op.isfile(f): os.remove(f)


class atdict(dict):
    'modified dictionary that works using ".key" '
    __getattr__= dict.__getitem__
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__


class Database(object):
    'Teslakit database'

    def __init__(self, data_folder):

        self.data_folder = data_folder
        self.paths = PathControl(self.data_folder)

        self.site_name = None

    def SetSite(self, site_name):
        'Sets current site'

        self.site_name = site_name
        p_site = op.join(self.paths.p_sites, site_name)

        # check site folder
        if not op.isdir(p_site):
            print('Teslakit Site not found at at {0}'.format(p_site))

        else:
            self.paths.SetSite(self.site_name)

    def MakeNewSite(self, site_name):
        'Makes all directories and .ini files needed for a new teslakit project site'

        # TODO REVISAR / UPDATEAR

        self.site_name = site_name
        p_site = op.join(self.paths.p_sites, site_name)

        # make site folder
        if not op.isdir(p_site):
            os.makedirs(p_site)

        else:
            print('Teslakit Site already exists at {0}'.format(p_site))
            self.paths.SetSite(self.site_name)
            return

        # copy site.ini template
        for fn in ['site.ini']:  #, 'parameters.ini']:
            copyfile(op.join(self.paths.p_resources, fn), op.join(p_site, fn))

        # create site subfolders
        self.paths.SetSite(self.site_name)
        for sf in self.paths.site.keys():
            p_sf = op.join(self.paths.p_site, sf)
            if not op.isdir(p_sf): os.makedirs(p_sf)

        # create export figs subfolders
        for k in self.paths.site.export_figs.keys():
            p_sf = self.paths.site.export_figs[k]
            if not op.isdir(p_sf): os.makedirs(p_sf)

        print('Teslakit Site generated at {0}'.format(p_site))

    def CheckInputFiles(self):
        'Checks teslakit required input files availability'

        # TODO REVISAR / UPDATEAR

        # files/folder to check ()
        conf = [
            ('MJO', ['hist'], [op.isfile]),
            ('TCs', ['noaa', 'nakajo_mats'], [op.isfile, op.isdir]),
            ('SST', ['hist_pacific'], [op.isfile]),
            ('WAVES', ['partitions_p1'], [op.isfile]),
            ('ESTELA', ['coastmat', 'estelamat', 'gowpoint', 'slp'],
             [op.isfile, op.isfile, op.isfile, op.isfile]),
            ('TIDE', ['mareografo_nc', 'hist_astro'], [op.isfile, op.isfile]),
        ]

        # get status
        l_status = []
        for k1, l_k2, l_fc in conf:
            for k2, fc in zip(l_k2, l_fc):
                fp = self.paths.site[k1][k2]
                rp = op.sep.join(fp.split(op.sep)[-4:])
                l_status.append(
                    ('{0}.{1}'.format(k1,k2), fc(fp), rp)
                )

        # print status with PrettyTable library
        tb = PrettyTable()
        tb.field_names = ["database ID", "available", "site path"]

        tb.align["database ID"] = "l"
        tb.align["available"] = "l"
        tb.align["site path"] = "l"

        for did, av, ap in l_status:
            tb.add_row([did, av, ap])

        print(tb)

    # database i/o

    # variables attrs (resources)
    def fill_metadata(self, xds, set_source=False):
        '''
        for each variable in xarray.Dataset xds, attributes will be set
        using resources/variables_attrs.json

        set_source - True for adding package source and institution metadata
        '''

        # read attributes dictionary
        p_vats = op.join(self.paths.p_resources, 'variables_attrs.json')
        with open(p_vats) as jf:
            d_vats = json.load(jf)

        # update dataset variables (names, units, descriptions)
        for vn in xds.variables:
            if vn.lower() in d_vats.keys():
               xds[vn].attrs = d_vats[vn.lower()]

        # set global attributes (source, institution)
        if set_source:
            xds.attrs['source'] = 'teslakit_v{0}'.format(__version__)
            #xds.attrs['institution'] = '{0}'.format(__author__)

        return xds

    def save_nc(self, xds, p_save, safe_time=False):
        '''
        (SECURE) exports xarray.Dataset to netcdf file format.

         - fills dataset with teslakit variables and source metadata
         - avoids overwritting problems
         - set safe_time=True if dataset contains dates beyond 2262
        '''

        # add teslakit metadata to xarray.Dataset
        xds = self.fill_metadata(xds, set_source=True)

        # remove previous file to avoid problems
        clean_files([p_save])

        # export .nc
        if safe_time:
            StoreBugXdset(xds, p_save)  # time dimension safe
        else:
            xds.to_netcdf(p_save, 'w')

    # SST

    def Load_SST(self):
        xds = xr.open_dataset(self.paths.site.SST.hist_pacific)
        xds = self.fill_metadata(xds)
        return xds

    def Save_SST_PCA(self, xds):
        self.save_nc(xds, self.paths.site.SST.pca)

    def Save_SST_KMA(self, xds):
        self.save_nc(xds, self.paths.site.SST.kma)

    def Save_SST_PCs_fit_rnd(self, d_PCs_fit, d_PCs_rnd):

        with open(self.paths.site.SST.d_pcs_fit, 'wb') as f:
            pickle.dump(d_PCs_fit, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.paths.site.SST.d_pcs_rnd, 'wb') as f:
            pickle.dump(d_PCs_rnd, f, protocol=pickle.HIGHEST_PROTOCOL)

    def Save_SST_AWT_sim(self, xds):
        self.save_nc(xds, self.paths.site.SST.awt_sim, safe_time=True)

    def Save_SST_PCs_sim(self, xds):

        # store yearly data
        self.save_nc(xds, self.paths.site.SST.pcs_sim, safe_time=True)

        # resample to daily and store
        xds_d = xds_reindex_daily(xds)
        self.save_nc(xds_d, self.paths.site.SST.pcs_sim_d, safe_time=True)

        # resample to monthly and store
        xds_m = xds_reindex_monthly(xds)
        self.save_nc(xds_m, self.paths.site.SST.pcs_sim_m, safe_time=True)

    def Load_SST_PCA(self):
        return xr.open_dataset(self.paths.site.SST.pca)

    def Load_SST_KMA(self):
        return xr.open_dataset(self.paths.site.SST.kma)

    def Load_SST_PCs_fit_rnd(self):

        with open(self.paths.site.SST.d_pcs_fit, 'rb') as f:
            d_PCs_fit = pickle.load(f)

        with open(self.paths.site.SST.d_pcs_rnd, 'rb') as f:
            d_PCs_rnd = pickle.load(f)

        return d_PCs_fit, d_PCs_rnd

    def Load_SST_AWT_sim(self):
        return xr.open_dataset(self.paths.site.SST.awt_sim, decode_times=True)

    def Load_SST_PCs_sim_d(self):
        return xr.open_dataset(self.paths.site.SST.pcs_sim_d, decode_times=True)

    def Load_SST_PCs_sim_m(self):
        return xr.open_dataset(self.paths.site.SST.pcs_sim_m, decode_times=True)

    # MJO

    def Load_MJO_hist(self):
        xds = xr.open_dataset(self.paths.site.MJO.hist)
        xds = self.fill_metadata(xds)
        return xds

    def Save_MJO_sim(self, xds):
        self.save_nc(xds, self.paths.site.MJO.sim, safe_time=True)

    def Load_MJO_sim(self):
        return xr.open_dataset(self.paths.site.MJO.sim, decode_times=True)

    # TCs

    def Load_TCs_noaa(self):
        return xr.open_dataset(self.paths.site.TCs.noaa)

    def Save_TCs_r1_hist(self, xds_tcs, xds_params):
        self.save_nc(xds_tcs, self.paths.site.TCs.hist_r1)
        self.save_nc(xds_params, self.paths.site.TCs.hist_r1_params)

    def Save_TCs_r2_hist(self, xds_tcs, xds_params):
        self.save_nc(xds_tcs, self.paths.site.TCs.hist_r2)
        self.save_nc(xds_params, self.paths.site.TCs.hist_r2_params)

    def Load_TCs_r1_hist(self):
        return xr.open_dataset(self.paths.site.TCs.hist_r1), \
                xr.open_dataset(self.paths.site.TCs.hist_r1_params)

    def Load_TCs_r2_hist(self):
        return xr.open_dataset(self.paths.site.TCs.hist_r2), \
                xr.open_dataset(self.paths.site.TCs.hist_r2_params)

    def Save_TCs_r1_sim_params(self, xds):
        self.save_nc(xds, self.paths.site.TCs.sim_r1_params)

    def Save_TCs_r2_sim_params(self, xds):
        self.save_nc(xds, self.paths.site.TCs.sim_r2_params)

    def Save_TCs_r1_mda_params(self, xds):
        self.save_nc(xds, self.paths.site.TCs.mda_r1_params)

    def Save_TCs_r2_mda_params(self, xds):
        self.save_nc(xds, self.paths.site.TCs.mda_r2_params)

    def Load_TCs_r2_mda_params(self):
        return xr.open_dataset(self.paths.site.TCs.mda_r2_params)

    def Load_TCs_r2_sim_params(self):
        return xr.open_dataset(self.paths.site.TCs.sim_r2_params)

    def Load_TCs_r2_mda_Simulations(self):
        return ReadTCsSimulations(self.paths.site.TCs.mda_r2_simulations)

    def Save_TCs_sim_r2_rbf_output(self, xds):
        self.save_nc(xds, self.paths.site.TCs.sim_r2_rbf_output)

    def Load_TCs_sim_r2_rbf_output(self):
        return xr.open_dataset(self.paths.site.TCs.sim_r2_rbf_output)

    def Load_TCs_Nakajo(self):
        return ReadNakajoMats(self.paths.site.TCs.nakajo_mats)

    def Save_TCs_probs_synth(self, xds):
        self.save_nc(xds, self.paths.site.TCs.probs_synth)

    def Load_TCs_probs_synth(self):
        return xr.open_dataset(self.paths.site.TCs.probs_synth)

    # WAVES

    def Load_WAVES_partitions(self):
        xds = ReadGowMat(self.paths.site.WAVES.partitions_p1)
        xds = self.fill_metadata(xds)
        return xds

    def Load_WAVES_partitions_nc(self):
        return xr.open_dataset(self.paths.site.WAVES.partitions_p1)

    def Save_WAVES_hist(self, xds):
        self.save_nc(xds, self.paths.site.WAVES.hist)

    def Load_WAVES_hist(self):
        return xr.open_dataset(self.paths.site.WAVES.hist)

    # ESTELA

    def Load_ESTELA_coast(self):
        return ReadCoastMat(self.paths.site.ESTELA.coastmat)

    def Load_ESTELA_data(self):
        return ReadEstelaMat(self.paths.site.ESTELA.estelamat)

    # TODO: remove?
    #def Load_ESTELA_waves_np(self):
    #    npzfile = np.load(self.paths.site.ESTELA.gowpoint)
    #    xr1 = xr.Dataset(
    #        {
    #            'Hs': (['time'], npzfile['Hs']),
    #            'Tp': (['time'], npzfile['Tp']),
    #            'Tm': (['time'], npzfile['Tm02']),
    #            'Dir': (['time'], npzfile['Dir']),
    #            'dspr': (['time'], npzfile['dspr'])
    #        },
    #        coords = {'time': npzfile['time']}
    #    )
    #    return  xr1

    def Load_ESTELA_SLP(self):
        return xr.open_dataset(self.paths.site.ESTELA.slp)

    def Load_ESTELA_KMA(self):
        p_est_kma = op.join(self.paths.site.ESTELA.pred_slp, 'kma.nc')
        return xr.open_dataset(p_est_kma)

    def Save_ESTELA_DWT_sim(self, xds):
        self.save_nc(xds, self.paths.site.ESTELA.sim_dwt, True)

    def Load_ESTELA_DWT_sim(self):
        return xr.open_dataset(
            self.paths.site.ESTELA.sim_dwt, decode_times=True)

    # HYDROGRAMS

    def Save_MU_TAU_hydrograms(self, l_xds):

        p_mutau = self.paths.site.ESTELA.hydrog_mutau
        if not op.isdir(p_mutau): os.makedirs(p_mutau)

        for x in l_xds:
            n_store = 'MUTAU_WT{0:02}.nc'.format(x.WT)
            self.save_nc(x, op.join(p_mutau, n_store))

    def Load_MU_TAU_hydrograms(self):

        p_mutau = self.paths.site.ESTELA.hydrog_mutau

        # MU - TAU intradaily hidrographs for each WWT
        l_mutau_ncs = sorted(
            [op.join(p_mutau, pf) for pf in os.listdir(p_mutau) if pf.endswith('.nc')]
        )
        l_xds = [xr.open_dataset(x) for x in l_mutau_ncs]
        return l_xds

    # TIDE

    def Load_TIDE_hist(self):
        xds_ml = xr.open_dataset(self.paths.site.TIDE.mareografo_nc)
        xds_at = xr.open_dataset(self.paths.site.TIDE.hist_astro)

        xds_ml = self.fill_metadata(xds_ml)
        xds_at = self.fill_metadata(xds_at)

        return xds_ml, xds_at

    def Save_TIDE_sim_astro(self, xds):
        self.save_nc(xds, self.paths.site.TIDE.sim_astro, True)

    def Load_TIDE_sim_astro(self):
        xds = xr.open_dataset(self.paths.site.TIDE.sim_astro, decode_times=True)

        # manual fix problems with hourly time
        d1 = xds.time.values[0]
        d2 = d1 + timedelta(hours=len(xds.time.values[:])-1)
        time_fix =  generate_datetimes(d1, d2, 'datetime64[h]')
        xds['time'] = time_fix

        return xds

    def Save_TIDE_hist_mmsl(self, xds):
        self.save_nc(xds, self.paths.site.TIDE.hist_mmsl)

    def Save_TIDE_sim_mmsl(self, xds):
        self.save_nc(xds, self.paths.site.TIDE.sim_mmsl, True)

    def Load_TIDE_hist_mmsl(self):
        xds = xr.open_dataset(self.paths.site.TIDE.hist_mmsl)
        return xds

    def Load_TIDE_sim_mmsl(self):
        xds = xr.open_dataset(self.paths.site.TIDE.sim_mmsl, decode_times=True)
        return xds


    # COMPLETE DATA 
    # TODO: continuar repaso

    def Load_HIST_Complete(self):
        '''
        Load all historical variables (hourly/3hourly):
            AWTs, DWTs, MJO, MMSL, AT, Hs, Tp, Dir, SS
        '''
        # load data
        AWT = self.Load_SST_KMA()
        MSL = self.Load_TIDE_hist_mmsl() # mmsl (mm)
        MJO = self.Load_MJO_hist()
        DWT = self.Load_ESTELA_KMA()  # bmus + 1
        ATD_h = self.Load_TIDE_hist_astro()
        WVS = self.Load_WAVES_hist()

        # data format
        AWT = xr.Dataset(
            {
                'bmus': AWT.bmus,
            },
            coords = {'time': AWT.time}
        )
        DWT = xr.Dataset(
            {
                'bmus': (('time',), DWT.bmus + 1),
            },
            coords = {'time': DWT.time.values[:]}
        )

        # get MJO categories 
        from .mjo import MJO_Categories
        mjo_cs, _ = MJO_Categories(MJO['rmm1'], MJO['rmm2'], MJO['phase'])
        MJO['bmus'] = (('time',), mjo_cs)

        # TODO ?
        # Hs, Tp, Dir from Aggregate_WavesFamilies
        #from .waves import Aggregate_WavesFamilies
        #WVS = Aggregate_WavesFamilies(WVS)

        # TODO SS?

        # reindex data to hourly (pad)
        AWT_h = fast_reindex_hourly(AWT)
        MSL_h = MSL.resample(time='1h').pad()
        MJO_h = fast_reindex_hourly(MJO)
        DWT_h = fast_reindex_hourly(DWT)
        WVS_h = fast_reindex_hourly(WVS)

        # common dates limits
        d1, d2 = xds_limit_dates(
            [AWT_h, ATD_h, MSL_h, MJO_h, DWT_h, ATD_h, WVS_h]
        )
        AWT_h = AWT_h.sel(time = slice(d1,d2))
        MSL_h = MSL_h.sel(time = slice(d1,d2))
        MJO_h = MJO_h.sel(time = slice(d1,d2))
        DWT_h = DWT_h.sel(time = slice(d1,d2))
        ATD_h = ATD_h.sel(time = slice(d1,d2))
        WVS_h = WVS_h.sel(time = slice(d1,d2))

        # copy to new dataset
        times = AWT_h.time.values[:]
        xds = xr.Dataset(
            {
                'AWT': (('time',), AWT_h.bmus.values[:].astype(int)),
                'MJO': (('time',), MJO_h.bmus.values[:].astype(int)),
                'DWT': (('time',), DWT_h.bmus.values[:].astype(int)),
                'MMSL': (('time',), MSL_h.mmsl.values[:] / 1000), # mm to m
                'AT': (('time',), ATD_h.tide.values[:]),
                'Hs': (('time',), WVS_h.Hs.values[:]),
                'Tp': (('time',), WVS_h.Tp.values[:]),
                'Dir': (('time',), WVS_h.Dir.values[:]),
            },
            coords = {'time': times}
        )

        return xds

    def Load_SIM_Covariates(self, n_sim_awt=0, n_sim_mjo=0, n_sim_dwt=0,
                            regenerate=False):
        '''
        Load all simulated covariates (hourly):
            AWTs, DWTs, MJO, MMSL, AT

        regenerate  - forces hourly dataset regeneration
        '''

        pf = self.paths.site.SIMULATION.covariates_hourly

        if not op.isfile(pf) or regenerate:
            xds = self.Generate_SIM_Covariates(n_sim_awt, n_sim_mjo, n_sim_dwt)
            self.save_nc(xds, pf, safe_time=True)

        else:
            xds = xr.open_dataset(pf, decode_times=True)

        return xds

    def Generate_SIM_Covariates(self, n_sim_awt=0, n_sim_mjo=0, n_sim_dwt=0):

        # load data
        AWT = self.Load_SST_AWT_sim()
        MSL = self.Load_TIDE_sim_mmsl()
        MJO = self.Load_MJO_sim()
        DWT = self.Load_ESTELA_DWT_sim()
        ATD_h = self.Load_TIDE_sim_astro()  # hourly data, 1 sim

        # select n_sim
        AWT = AWT.isel(n_sim=n_sim_awt)
        MSL = MSL.isel(n_sim=n_sim_awt)
        MJO = MJO.isel(n_sim=n_sim_mjo)
        DWT = DWT.isel(n_sim=n_sim_dwt)

        # reindex data to hourly (pad)
        AWT_h = fast_reindex_hourly(AWT)
        MSL_h = fast_reindex_hourly(MSL)
        MJO_h = fast_reindex_hourly(MJO)
        DWT_h = fast_reindex_hourly(DWT)

        # common dates limits
        d1, d2 = xds_limit_dates([AWT_h, ATD_h, MSL_h, MJO_h, DWT_h, ATD_h])
        AWT_h = AWT_h.sel(time = slice(d1,d2))
        MSL_h = MSL_h.sel(time = slice(d1,d2))
        MJO_h = MJO_h.sel(time = slice(d1,d2))
        DWT_h = DWT_h.sel(time = slice(d1,d2))
        ATD_h = ATD_h.sel(time = slice(d1,d2))

        # copy to new dataset
        times = AWT_h.time.values[:]
        xds = xr.Dataset(
            {
                'AWT': (('time',), AWT_h.evbmus_sims.values[:].astype(int)),
                'MJO': (('time',), MJO_h.evbmus_sims.values[:].astype(int)),
                'DWT': (('time',), DWT_h.evbmus_sims.values[:].astype(int)),
                'MMSL': (('time',), MSL_h.mmsl.values[:]),
                'AT': (('time',), ATD_h.astro.values[:]),
            },
            coords = {'time': times}
        )

        return xds

    def Save_SIM_Waves(self, xds):
        self.save_nc(xds, self.paths.site.SIMULATION.waves_hourly,
                     safe_time=True)

    def Load_SIM_Waves(self):
        return xr.open_dataset(
            self.paths.site.SIMULATION.waves_hourly, decode_times=True)

    def Save_SIM_Complete(self, xds):
        self.save_nc(xds, self.paths.site.SIMULATION.complete_hourly,
                     safe_time=True)

    def Load_SIM_Complete(self):
        return xr.open_dataset(
            self.paths.site.SIMULATION.complete_hourly, decode_times=True)

    # SPECIAL PLOTS

    def Load_AWTs_DWTs_Plots_sim(self, n_sim=0):
        'Load data needed for WT-WT Probs plot'

        # simulated
        xds_AWT = self.Load_SST_AWT_sim()
        xds_DWT = self.Load_ESTELA_DWT_sim()

        # AWT simulated - evbmus_sims -1 
        xds_AWT = xr.Dataset(
            {'bmus': (('time',), xds_AWT.evbmus_sims.isel(n_sim=n_sim)-1)},
            coords = {'time': xds_AWT.time.values[:]}
        )

        # DWT simulated - evbmus_sims -1
        xds_DWT = xr.Dataset(
            {'bmus': (('time',), xds_DWT.evbmus_sims.isel(n_sim=n_sim)-1)},
            coords = {'time': xds_DWT.time.values[:]}
        )

        # reindex AWT to daily dates (year pad to days)
        xds_AWT = xds_reindex_daily(xds_AWT)

        # get common dates
        dc = xds_common_dates_daily([xds_AWT, xds_DWT])
        xds_DWT = xds_DWT.sel(time=slice(dc[0], dc[-1]))
        xds_AWT = xds_AWT.sel(time=slice(dc[0], dc[-1]))

        return xds_AWT, xds_DWT

    def Load_AWTs_DWTs_Plots_hist(self):
        'Load data needed for WT-WT Probs plot'

        # historical
        xds_AWT = self.Load_SST_KMA()
        xds_DWT = self.Load_ESTELA_KMA()

        # AWT historical - bmus
        xds_AWT = xr.Dataset(
            {'bmus': (('time',), xds_AWT.bmus.values[:])},
            coords = {'time': xds_AWT.time.values[:]}
        )

        # DWT historical - sorted_bmus_storms
        xds_DWT = xr.Dataset(
            {'bmus': (('time',), xds_DWT.sorted_bmus_storms.values[:])},
            coords = {'time': xds_DWT.time.values[:]}
        )

        # reindex AWT to daily dates (year pad to days)
        xds_AWT = xds_reindex_daily(xds_AWT)

        # get common dates
        dc = xds_common_dates_daily([xds_AWT, xds_DWT])
        xds_DWT = xds_DWT.sel(time=slice(dc[0], dc[-1]))
        xds_AWT = xds_AWT.sel(time=slice(dc[0], dc[-1]))

        return xds_AWT, xds_DWT

    def Load_MJO_DWTs_Plots_hist(self):
        'Load data needed for WT-WT Probs plot'

        # historical
        xds_MJO = self.Load_MJO_hist()
        xds_DWT = self.Load_ESTELA_KMA()

        # MJO historical - phase -1
        xds_MJO['phase'] = xds_MJO.phase - 1

        # DWT historical - sorted_bmus
        xds_DWT = xr.Dataset(
            {'bmus': (('time',), xds_DWT.sorted_bmus.values[:])},
            coords = {'time': xds_DWT.time.values[:]}
        )

        # get common dates
        dc = xds_common_dates_daily([xds_DWT, xds_MJO])
        xds_DWT = xds_DWT.sel(time=slice(dc[0], dc[-1]))
        xds_MJO = xds_MJO.sel(time=slice(dc[0], dc[-1]))

        return xds_MJO, xds_DWT

    def Load_MJO_DWTs_Plots_sim(self, n_sim=0):
        'Load data needed for WT-WT Probs plot'

        # simulated
        xds_MJO = self.Load_MJO_sim()
        xds_DWT = self.Load_ESTELA_DWT_sim()

        # DWT simulated - evbmus_sims
        xds_DWT = xr.Dataset(
            {'bmus': (('time',), xds_DWT.evbmus_sims.isel(n_sim=n_sim))},
            coords = {'time': xds_DWT.time.values[:]}
        )

        # get common dates
        dc = xds_common_dates_daily([xds_DWT, xds_MJO])
        xds_DWT = xds_DWT.sel(time=slice(dc[0], dc[-1]))
        xds_MJO = xds_MJO.sel(time=slice(dc[0], dc[-1]))

        return xds_MJO, xds_DWT

    # NEARSHORE: COMPLETE DATASETS (RBF DATASETS)

    def Save_NEARSHORE_FULL_sea(self, pd_waves):
        'Stores sea waves full dataset. Used at RBF Reconstruction'

        pd_waves.to_pickle(self.paths.site.NEARSHORE.sea_dataset)

    def Load_NEARSHORE_FULL_sea(self):
        'Load sea waves full dataset'

        return pd.read_pickle(self.paths.site.NEARSHORE.sea_dataset)

    def Save_NEARSHORE_FULL_swell(self, pd_waves):
        'Stores swells waves full dataset. Used at RBF Reconstruction'

        pd_waves.to_pickle(self.paths.site.NEARSHORE.swl_dataset)

    def Load_NEARSHORE_FULL_swell(self):
        'Load swells waves full datasets'

        return pd.read_pickle(self.paths.site.NEARSHORE.swl_dataset)

    # NEARSHORE: MDA CLASSIFICATION (RBF SUBSET)

    def Save_NEARSHORE_MDA_sea(self, pd_waves):
        'Stores sea waves subset (from mda selection)'

        pd_waves.to_pickle(self.paths.site.NEARSHORE.sea_subset)

    def Load_NEARSHORE_MDA_sea(self):
        'Load sea waves subset (from mda selection)'

        return pd.read_pickle(self.paths.site.NEARSHORE.sea_subset)

    def Save_NEARSHORE_MDA_swell(self, pd_waves):
        'Stores swell waves subset (from mda selection)'

        pd_waves.to_pickle(self.paths.site.NEARSHORE.swl_subset)

    def Load_NEARSHORE_MDA_swell(self):
        'Load swell waves subset (from mda selection)'

        return pd.read_pickle(self.paths.site.NEARSHORE.swl_subset)

    # NEARSHORE: POINT PROPAGATION (RBF TARGET)

    def Save_NEARSHORE_TARGET_sea(self, pd_waves):
        'Stores sea waves target (from swan propagation)'

        pd_waves.to_pickle(self.paths.site.NEARSHORE.sea_target)

    def Load_NEARSHORE_TARGET_sea(self):
        'Load sea waves target (from swan propagation)'

        return pd.read_pickle(self.paths.site.NEARSHORE.sea_target)

    def Save_NEARSHORE_TARGET_swell(self, pd_waves):
        'Stores swell waves target (from swan propagation)'

        pd_waves.to_pickle(self.paths.site.NEARSHORE.swl_target)

    def Load_NEARSHORE_TARGET_swell(self):
        'Load swell waves target (from swan propagation)'

        return pd.read_pickle(self.paths.site.NEARSHORE.swl_target)

    # NEARSHORE: RBF RECONSTRUCTION

    def Save_NEARSHORE_RECONSTRUCTION_sea(self, pd_waves):
        'Stores sea waves RBF reconstruction'

        pd_waves.to_pickle(self.paths.site.NEARSHORE.sea_recon)

    def Load_NEARSHORE_RECONSTRUCTION_sea(self):
        'Load sea waves RBF reconstruction'

        return pd.read_pickle(self.paths.site.NEARSHORE.sea_recon)

    def Save_NEARSHORE_RECONSTRUCTION_swell(self, pd_waves):
        'Stores swell waves RBF reconstruction'

        pd_waves.to_pickle(self.paths.site.NEARSHORE.swl_recon)

    def Load_NEARSHORE_RECONSTRUCTION_swell(self):
        'Load swell waves RBF reconstruction'

        return pd.read_pickle(self.paths.site.NEARSHORE.swl_recon)

class PathControl(object):
    'auxiliar object for handling teslakit files paths'

    def __init__(self, p_data):

        # ini templates
        p_resources = op.join(op.dirname(op.realpath(__file__)), 'resources')

        # teslakit sites
        p_sites = op.join(p_data, 'sites')

        # string paths
        self.p_data = p_data
        self.p_resources = p_resources
        self.p_sites = p_sites

        self.p_site = None  # current site

        # atdicts
        self.site = None

    def __str__(self):
        'Print paths'
        # TODO: update

        txt = ''
        if isinstance(self.site, dict):
            txt+= '\n\nSite Files:'
            for k1 in sorted(self.site.keys()):
                for k2 in sorted(self.site[k1].keys()):
                    aux1 ='\n.site.{0}.{1}'.format(k1, k2)
                    aux2 = self.site[k1][k2]
                    if 'teslakit' in aux2:
                        aux2 = aux2.split('teslakit')[1]
                    txt+='{0:.<45} {1}'.format(aux1, aux2)
        return txt

    def FilesSummary(self):
        'return detailed database and site files summary'
        # TODO: update

        txt = ''
        if isinstance(self.site,dict):
            txt+= '\n\nSite Files:'
            for k1 in sorted(self.site.keys()):
                for k2 in sorted(self.site[k1].keys()):
                    txt += description(self.site[k1][k2])

        return txt

    def SetSite(self, site_name):
        'Sets dictionary with site files'

        # site folder 
        p_site = op.join(self.p_sites, site_name)
        p_ini = op.join(p_site, 'site.ini')

        # use configparser lib
        cfg = configparser.ConfigParser()
        cfg.read(p_ini)

        dd = {}
        for k in cfg.sections():
            ss = atdict(cfg[k])
            for kk in ss.keys():
                if not ss[kk].startswith(os.sep) and not ':\\' in ss[kk]:
                    ss[kk] = op.join(p_site, k, ss[kk])
            dd[k] = atdict(ss)

        self.site = atdict(dd)
        self.p_site = p_site

