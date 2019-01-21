#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op
import configparser

from lib.io.getinfo import description

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
            print ''
            for k2 in sorted(self.params[k1].keys()):
                aux ='\n.params.{0}.{1}'.format(k1, k2)
                txt+='{0:.<45} {1}'.format(aux, self.params[k1][k2])
        print(txt)


class PathControl(object):
    'auxiliar object for handling database and site paths'
    # TODO: COMMON AND SITE DATABASE CAN BE IMPROVED

    def __init__(self):

        # teslakit data
        p_data = op.join(os.sep.join(op.realpath(__file__).split(op.sep)[0:-3]),'data')

        # teslakit database and sites
        p_DB = op.join(p_data, 'database')
        p_DB_ini = op.join(p_data, 'database.ini')
        p_sites = op.join(p_data, 'sites')

        # string paths
        self.p_data = p_data
        self.p_DB = p_DB
        self.p_DB_ini = p_DB_ini
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
                aux1 ='\n.DB.{0}.{1}'.format(k1, k2)
                aux2 = self.DB[k1][k2]
                if 'teslakit' in aux2:
                    aux2 = aux2.split('teslakit')[1]
                txt+='{0:.<45} {1}'.format(aux1, aux2)

        if isinstance(self.site,dict):
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

        txt = ''
        txt+= '\nDatabase Files:'
        for k1 in sorted(self.DB.keys()):
            for k2 in sorted(self.DB[k1].keys()):
                txt += description(self.DB[k1][k2])

        if isinstance(self.site,dict):
            txt+= '\n\nSite Files:'
            for k1 in sorted(self.site.keys()):
                for k2 in sorted(self.site[k1].keys()):
                    txt += description(self.site[k1][k2])

        return txt

    def SetDatabase(self):
        'Set database paths from database.ini file'


        p_DB = self.p_DB
        p_ini = self.p_DB_ini

        # use configparser lib
        cfg = configparser.ConfigParser()
        cfg.read(p_ini)

        dd = {}
        for k in cfg.sections():
            ss = atdict(cfg[k])
            for kk in ss.keys():
                if not ss[kk].startswith(os.sep) and not ':\\' in ss[kk]:
                    ss[kk] = op.join(p_DB, k, ss[kk])
            dd[k] = atdict(ss)

        self.DB = atdict(dd)

    def SetSite(self, site_name):
        'Sets dictionary with site files'

        # site folder 
        p_site = op.join(self.p_sites, site_name)
        p_ini = op.join(p_site, 'files.ini')

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
        self.p_site_ini = op.join(p_site, 'site.ini')

