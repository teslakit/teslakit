#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import configparser
from shutil import copyfile
import pickle

from prettytable import PrettyTable
import xarray as xr

from .io.getinfo import description
from .io.aux_nc import StoreBugXdset
from .io.matlab import ReadTCsSimulations, ReadMatfile, ReadNakajoMats, \
ReadGowMat, ReadCoastMat, ReadEstelaMat
from .custom_dateutils import xds_reindex_daily
from .custom_dateutils import xds_reindex_monthly


class atdict(dict):
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
    def Load_SST(self):
        return xr.open_dataset(self.paths.site.SST.hist_pacific)

    def Save_SST_PCA(self, xds):
        xds.to_netcdf(self.paths.site.SST.pca, 'w')

    def Save_SST_KMA(self, xds):
        xds.to_netcdf(self.paths.site.SST.kma, 'w')

    def Save_SST_PCs_fit_rnd(self, d_PCs_fit, d_PCs_rnd):

        with open(self.paths.site.SST.d_pcs_fit, 'wb') as f:
            pickle.dump(d_PCs_fit, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.paths.site.SST.d_pcs_rnd, 'wb') as f:
            pickle.dump(d_PCs_rnd, f, protocol=pickle.HIGHEST_PROTOCOL)

    def Save_SST_PCs_sim(self, xds):

        # store yearly data
        StoreBugXdset(xds, self.paths.site.SST.pcs_sim)

        # resample to daily and store
        xds_d = xds_reindex_daily(xds)
        StoreBugXdset(xds_d, self.paths.site.SST.pcs_sim_d)

        # resample to monthly and store
        xds_m = xds_reindex_monthly(xds)
        StoreBugXdset(xds_m, self.paths.site.SST.pcs_sim_m)

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

    def Load_SST_PCs_sim_d(self):
        return xr.open_dataset(self.paths.site.SST.pcs_sim_d)

    def Load_SST_PCs_sim_m(self):
        return xr.open_dataset(self.paths.site.SST.pcs_sim_m)

    def Load_MJO_hist(self):
        return xr.open_dataset(self.paths.site.MJO.hist)

    def Save_MJO_sim(self, xds):
        StoreBugXdset(xds, self.paths.site.MJO.sim)

    def Load_MJO_sim(self):
        return xr.open_dataset(self.paths.site.MJO.sim)

    def Load_TCs_noaa(self):
        return xr.open_dataset(self.paths.site.TCs.noaa)

    # TODO: indicar _hist en los 4 siguientes
    def Save_TCs_r1(self, xds_tcs, xds_params):
        xds_tcs.to_netcdf(self.paths.site.TCs.hist_r1, 'w')
        xds_params.to_netcdf(self.paths.site.TCs.hist_r1_params, 'w')

    def Save_TCs_r2(self, xds_tcs, xds_params):
        xds_tcs.to_netcdf(self.paths.site.TCs.hist_r2, 'w')
        xds_params.to_netcdf(self.paths.site.TCs.hist_r2_params, 'w')

    def Load_TCs_r1(self):
        return xr.open_dataset(self.paths.site.TCs.hist_r1), \
                xr.open_dataset(self.paths.site.TCs.hist_r1_params)

    def Load_TCs_r2(self):
        return xr.open_dataset(self.paths.site.TCs.hist_r2), \
                xr.open_dataset(self.paths.site.TCs.hist_r2_params)

    def Save_TCs_r1_sim_params(self, xds):
        xds.to_netcdf(self.paths.site.TCs.sim_r1_params, 'w')

    def Save_TCs_r2_sim_params(self, xds):
        xds.to_netcdf(self.paths.site.TCs.sim_r2_params, 'w')

    def Save_TCs_r1_mda_params(self, xds):
        xds.to_netcdf(self.paths.site.TCs.mda_r1_params, 'w')

    def Save_TCs_r2_mda_params(self, xds):
        xds.to_netcdf(self.paths.site.TCs.mda_r2_params, 'w')

    def Load_TCs_r2_mda_params(self):
        return xr.open_dataset(self.paths.site.TCs.mda_r2_params)

    def Load_TCs_r2_sim_params(self):
        return xr.open_dataset(self.paths.site.TCs.sim_r2_params)

    def Load_TCs_r2_mda_Simulations(self):
        return ReadTCsSimulations(self.paths.site.TCs.mda_r2_simulations)

    def Save_TCs_sim_r2_rbf_output(self, xds):
        xds.to_netcdf(self.paths.site.TCs.sim_r2_rbf_output, 'w')

    def Load_TCs_sim_r2_rbf_output(self):
        return xr.open_dataset(self.paths.site.TCs.sim_r2_rbf_output)

    def Load_TCs_Nakajo(self):
        return ReadNakajoMats(self.paths.site.TCs.nakajo_mats)

    def Save_TCs_probs_synth(self, xds):
        xds.to_netcdf(self.paths.site.TCs.probs_synth, 'w')

    def Load_TCs_probs_synth(self):
        return xr.open_dataset(self.paths.site.TCs.probs_synth)

    def Load_WAVES_partitions(self):
        return ReadGowMat(self.paths.site.WAVES.partitions_p1)

    def Load_WAVES_fams_noTCs(self):
        return xr.open_dataset(self.paths.site.WAVES.families_notcs)

    def Save_WAVES_ptsfams_noTCs(self, xds_pts, xds_fams):
        xds_pts.to_netcdf(self.paths.site.WAVES.partitions_notcs, 'w')
        xds_fams.to_netcdf(self.paths.site.WAVES.families_notcs, 'w')

    def Save_WAVES_fams_TCs_categ(self, d_fams):

        p_fams = self.paths.site.WAVES.families_tcs_categ

        if not op.isdir(p_fams): os.makedirs(p_fams)
        for k in d_fams.keys():
            p_s = op.join(p_fams, 'waves_fams_cat{0}.nc'.format(k))
            d_fams[k].to_netcdf(p_s, 'w')

    def Load_WAVES_fams_TCs_categ(self):

        p_fams = self.paths.site.WAVES.families_tcs_categ

        d_WT_TCs_wvs = {}
        for k in range(6):
            p_s = op.join(p_fams, 'waves_fams_cat{0}.nc'.format(k))
            d_WT_TCs_wvs['{0}'.format(k)] = xr.open_dataset(p_s)
        return d_WT_TCs_wvs

    def Load_ESTELA_coast(self):
        return ReadCoastMat(self.paths.site.ESTELA.coastmat)

    def Load_ESTELA_data(self):
        return ReadEstelaMat(self.paths.site.ESTELA.estelamat)

    def Load_ESTELA_waves(self):
        return ReadGowMat(self.paths.site.ESTELA.gowpoint)

    def Load_ESTELA_SLP(self):
        return xr.open_dataset(self.paths.site.ESTELA.slp)

    def Load_ESTELA_KMA(self):
        p_est_kma = op.join(self.paths.site.ESTELA.pred_slp, 'kma.nc')
        return xr.open_dataset(p_est_kma)

    def Save_ESTELA_DWT_sim(self, xds):
        StoreBugXdset(xds, self.paths.site.ESTELA.sim_dwt)

    def Load_ESTELA_DWT_sim(self):
        return xr.open_dataset(self.paths.site.ESTELA.sim_dwt)

    def Save_MU_TAU_hydrograms(self, l_xds):

        p_mutau = self.paths.site.ESTELA.hydrog_mutau

        if not op.isdir(p_mutau): os.makedirs(p_mutau)
        for x in l_xds:
            n_store = 'MUTAU_WT{0:02}.nc'.format(x.WT)
            x.to_netcdf(op.join(p_mutau, n_store), 'w')

    def Load_MU_TAU_hydrograms(self):

        p_mutau = self.paths.site.ESTELA.hydrog_mutau

        # MU - TAU intradaily hidrographs for each WWT
        l_mutau_ncs = sorted(
            [op.join(p_mutau, pf) for pf in os.listdir(p_mutau) if pf.endswith('.nc')]
        )
        l_xds = [xr.open_dataset(x) for x in l_mutau_ncs]
        return l_xds

    def Load_TIDE_hist_astro(self):
        xds = xr.open_dataset(self.paths.site.TIDE.hist_astro)
        xds.rename({'observed':'level','predicted':'tide'}, inplace=True)
        return xds

    def Save_TIDE_sim_astro(self, xds):
        StoreBugXdset(xds, self.paths.site.TIDE.sim_astro)

    def Save_TIDE_sim_mmsl(self, xds):
        StoreBugXdset(xds, self.paths.site.TIDE.sim_mmsl)

    def Load_TIDE_mareografo(self):
        xds = xr.open_dataset(self.paths.site.TIDE.mareografo_nc)

        # fix data
        xds.rename({'WaterLevel':'tide'}, inplace=True)
        xds['tide'] = xds['tide'] * 1000
        return xds


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

