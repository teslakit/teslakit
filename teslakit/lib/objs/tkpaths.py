#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import configparser

class atdict(dict):
    __getattr__= dict.__getitem__
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__


class Site(object):
    'Project site: collection of parameters and file paths'

    def __init__(self, site_name):
        self.name = site_name

        # path control
        self.pc = PathControl()
        self.pc.SetSite(site_name)

        # site parameters
        self.params = self.ReadParameters()

    def Summary(self):
        'Print Site info summary: params + file paths'

        self.PrintParameters()
        print(self.pc)

    def ReadParameters(self):
        'Read site parameters from site.ini file'

        p_ini = self.pc.p_site_ini

        # use configparser lib
        cfg = configparser.ConfigParser()
        cfg.read(p_ini)

        dd = {}
        for k in cfg.sections():
            dd[k] = atdict(cfg[k])
        return atdict(dd)

    def PrintParameters(self):
        'print site parameters from .ini file'

        txt='\nSite Parameters'
        for k1 in sorted(self.params.keys()):
            for k2 in sorted(self.params[k1].keys()):
                aux ='\n.params.{0}.{1}'.format(k1, k2)
                txt+='{0:30} = {1}'.format(aux, self.params[k1][k2])
        print(txt)


class PathControl(object):
    'auxiliar object for handling database and site paths'
    # TODO FUNCION QUE COMPRUEBE ARCHIVOS INPUT DE SITE SIN EJECUTAR

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

    def __str__(self):
        'Print paths'

        txt = ''
        txt+= '\nDatabase Files:'
        for k1 in sorted(self.DB.keys()):
            for k2 in sorted(self.DB[k1].keys()):
                aux ='\n.DB.{0}.{1}'.format(k1, k2)
                txt+='{0:27} - {1}'.format(aux, self.DB[k1][k2])

        if isinstance(self.site,dict):
            txt+= '\n\nSite Files:'
            for k1 in sorted(self.site.keys()):
                for k2 in sorted(self.site[k1].keys()):
                    aux ='\n.site.{0}.{1}'.format(k1, k2)
                    txt+='{0:27} - {1}'.format(aux, self.site[k1][k2])
        return txt

    def SetDatabase(self):
        'Set paths not related with a study site'

        # tropical cyclones 
        pt = op.join(self.p_DB, 'TCs')
        dd_tcs = {
            'nakajo_mats': op.join(pt, 'Nakajo_tracks'),
            'noaa':        op.join(pt, 'Allstorms.ibtracs_wmo.v03r10.nc'),
            'noaa_fix':    op.join(pt, 'Allstorms.ibtracs_wmo.v03r10_fix.nc'),
        }

        # MJO
        pt = op.join(self.p_DB, 'MJO')
        dd_mjo = {
            'hist': op.join(pt, 'MJO_hist.nc'),
        }

        # SST
        pt = op.join(self.p_DB, 'SST')
        dd_sst = {
            'hist_pacific' :op.join(pt, 'SST_1854_2017_Pacific.nc'),
        }

        # SLP
        pt = op.join(self.p_DB, 'CFS')
        dd_slp = {
            'cfs_prmsl': op.join(pt, 'prmsl'),
        }

        # RAW download
        pt = op.join(self.p_DB, 'download')
        dd_dwl = {
            'CSIRO':op.join(pt, 'CSIRO'),
        }

        # main database dict
        dd = {
            'tcs': atdict(dd_tcs),
            'mjo': atdict(dd_mjo),
            'sst': atdict(dd_sst),
            'slp': atdict(dd_slp),
            'sst': atdict(dd_sst),
            'dwl': atdict(dd_dwl),
        }
        self.DB = atdict(dd)

    def SetSite(self, site_name):
        'Sets dictionary with site files'

        # site folder 
        p_site = op.join(self.p_sites, site_name)

        # SST
        pt = op.join(p_site, 'SST')
        dd_sst = {
            'PCA': op.join(pt, 'SST_PCA.nc'),
            'KMA': op.join(pt, 'SST_KMA.nc'),
        }

        # mjo
        pt = op.join(p_site, 'MJO')
        dd_mjo = {
            'alrw': op.join(pt, 'alr_w'),       # auto regresive logistic sim 
            'sim':  op.join(pt, 'MJO_sim.nc'),  # sim file
        }

        # estela predictor
        pt = op.join(p_site, 'ESTELA')
        dd_estela = {
            'estelamat': op.join(pt, 'kwajalein_roi_obj.mat'),
            'coastmat':  op.join(pt, 'Costa.mat'),
            'gowpoint':  op.join(pt, 'gow2_062_ 9.50_167.25.mat'),
            'slp':       op.join(pt, 'SLP.nc'),
            'pred_slp':  op.join(pt, 'pred_SLP'),
        }

        # tropical cyclones
        pt = op.join(p_site, 'TCs')
        dd_tcs = {
            'circle_hist': op.join(pt, 'TCs_hist_circle.nc'),
            'probs_synth': op.join(pt, 'TCs_synth_ProbsChange.nc'),
        }

        # tide gauges 
        # TODO: RESOLVER DUDA DATOS MAREA. JUNTAR EN NC
        pt = op.join(p_site, 'TIDE')
        dd_tds = {
            'mareografo':  op.join(pt, 'Mareografo_KWA.mat'),
            'MAR_1820000': op.join(pt, 'MAR_1820000.mat'),
            'sim_astro': op.join(pt, 'tide_astro_sim.nc'),
        }

        # waves data 
        p_wvs_parts = op.join(p_site, 'WAVES', 'wave_partitions')
        p_wvs_procs = op.join(p_site, 'WAVES', 'wave_process')
        dd_waves = {

            # input
            'partitions': p_wvs_parts,
            'partitions_p1': op.join(p_wvs_parts, 'point1.mat'),

            # execution files
            'process': p_wvs_procs,
            'partitions_noTCs': op.join(p_wvs_procs, 'waves_partitions_noTCs.nc'),
            'families_noTCs':   op.join(p_wvs_procs, 'waves_families_noTCs.nc'),
        }

        # n-years simulation files 
        #dd_sim = {
        #    'mjo':op.join(p_site, 'SIM', 'MJO_sim.nc'),
        #}

        # export figs and reports
        pt = op.join(p_site, 'export_figs')
        dd_export = {
            'mjo': op.join(pt, 'mjo'),
            'sst': op.join(pt, 'sst'),
        }


        # main site dict
        dd = {
            'sst': atdict(dd_sst),
            'mjo': atdict(dd_mjo),
            'est': atdict(dd_estela),
            'tcs': atdict(dd_tcs),
            'tds': atdict(dd_tds),
            'wvs': atdict(dd_waves),
            #'sim': atdict(dd_sim),
            'exp': atdict(dd_export),
        }
        self.site = atdict(dd)
        self.p_site = p_site
        self.p_site_ini = op.join(p_site, 'site.ini')

