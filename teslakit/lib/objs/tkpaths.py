#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op

class atdict(dict):
    __getattr__= dict.__getitem__
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__

class PathControl(object):
    '''
    auxiliar object for handling database and site paths
    '''

    def __init__(self):

        # teslakit data
        p_data = op.join(os.sep.join(op.realpath(__file__).split(op.sep)[0:-3]),'data')

        # teslakit database and sites
        p_DB = op.join(p_data, 'database')
        p_sites = op.join(p_data, 'sites')

        # string paths
        self.p_data = p_data
        self.p_DB = p_DB
        self.p_sites = p_sites
        self.p_site = None  # current site

        # atdicts
        self.DB = None
        self.site = None

        # initialize 
        self.SetDatabase()

    def SetDatabase(self):
        'Set paths not related with a study site'

        # tropical cyclones 
        dd_tcs = {
            'nakajo':op.join(self.p_DB, 'TCs', 'Nakajo_tracks'),
            'noaa':op.join(self.p_DB, 'TCs','Allstorms.ibtracs_wmo.v03r10.nc'),
            'noaa_fix':op.join(self.p_DB, 'TCs','Allstorms.ibtracs_wmo.v03r10_fix.nc'),
        }

        # mjo
        dd_mjo = {
            'hist':op.join(self.p_DB, 'MJO', 'MJO_hist.nc'),
        }

        # SST
        dd_sst = {
            'hist_pacific':op.join(self.p_DB, 'SST','SST_1854_2017_Pacific.nc'),
        }

        # SLP
        dd_slp = {
            'cfs_prmsl':op.join(self.p_DB, 'CFS', 'prmsl'),
        }

        # main database dict
        dd = {
            'tcs': atdict(dd_tcs),
            'mjo': atdict(dd_mjo),
            'sst': atdict(dd_sst),
            'slp': atdict(dd_slp),
            'sst': atdict(dd_sst),
        }
        self.DB = atdict(dd)

    def SetSite(self, site_name):
        'Sets dictionary with site files'

        # site folder 
        p_site = op.join(self.p_sites, site_name)

        # SST
        dd_sst = {
            'PCA':op.join(p_site, 'SST', 'SST_PCA.nc'),
            'KMA':op.join(p_site, 'SST', 'SST_KMA.nc'),
        }

        # mjo
        dd_mjo = {
            'alrw':op.join(p_site, 'MJO', 'alr_w'),
            'sim':op.join(p_site, 'MJO', 'MJO_sim.nc'),
        }

        # estela predictor
        dd_estela = {
            'estelamat':op.join(p_site, 'estela', 'kwajalein_roi_obj.mat'),
            'coastmat':op.join(p_site, 'estela', 'Costa.mat'),
            'gowpoint':op.join(p_site, 'estela', 'gow2_062_ 9.50_167.25.mat'),
            'slp':op.join(p_site, 'estela', 'SLP.nc'),
            'pred_slp':op.join(p_site, 'estela', 'pred_SLP'),
        }

        # tropical cyclones
        dd_tcs = {
            'circle_hist':op.join(p_site, 'TCs', 'TCs_hist_circle.nc'),
        }

        # tide gauges 
        dd_tds = {
            'mareografo':op.join(p_site, 'tide', 'Mareografo_KWA.mat'),
            'MAR_1820000':op.join(p_site, 'tide', 'MAR_1820000.mat'),
        }

        # waves data 
        p_wvs_parts = op.join(p_site, 'wave_partitions')
        p_wvs_procs = op.join(p_site, 'wave_process')
        dd_waves = {

            # input
            'partitions': p_wvs_parts,
            'partitions_p1':op.join(p_wvs_parts, 'point1.mat'),

            # execution files
            'process': p_wvs_procs,
            'partitions_noTCs':op.join(p_wvs_procs,'waves_partitions_noTCs.nc'),
            'families_noTCs':op.join(p_wvs_procs, 'waves_families_noTCs.nc'),
        }

        # n-years simulation files 
        #dd_sim = {
        #    'mjo':op.join(p_site, 'SIM', 'MJO_sim.nc'),
        #}

        # main site dict
        dd = {
            'sst': atdict(dd_sst),
            'mjo': atdict(dd_mjo),
            'est': atdict(dd_estela),
            'tcs': atdict(dd_tcs),
            'tds': atdict(dd_tds),
            'wvs': atdict(dd_waves),
            #'sim': atdict(dd_sim),
            'export_figs':op.join(p_site, 'export_figs')
        }
        self.site = atdict(dd)
        self.p_site = p_site

    def __str__(self):
        'Print paths'

        txt = ''
        txt+= '\nDatabase:'
        for k1 in sorted(self.DB.keys()):
            for k2 in sorted(self.DB[k1].keys()):
                txt+='\n.DB.{0}.{1:15} - {2}'.format(k1, k2, self.DB[k1][k2])

        if isinstance(self.site,dict):
            txt+= '\n\nSite:'
            for k1 in sorted(self.site.keys()):
                for k2 in sorted(self.site[k1].keys()):
                    txt+='\n.site.{0}.{1:13} - {2}'.format(k1, k2, self.site[k1][k2])
        return txt[1:]

