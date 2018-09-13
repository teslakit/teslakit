#!/usr/bin/env python
# -*- coding: utf-8 -*-

# basic import
import os
import os.path as op

class PathControl(object):
    '''
    auxiliar object for handling database and site paths
    '''

    def __init__(self, p_data):

        # database paths
        self.p_data = p_data

        self.p_db_storms = op.join(p_data, 'STORMS')
        self.p_db_nakajo_mats = op.join(self.p_db_storms, 'Nakajo_tracks')
        self.p_db_NOAA = op.join(self.p_db_storms, 'Allstorms.ibtracs_wmo.v03r10.nc')
        self.p_db_NOAA_fix = op.join(self.p_db_storms, 'Allstorms.ibtracs_wmo.v03r10_fix.nc')


        # site paths
        self.p_st_storms_hist_circle = op.join(self.p_db_storms, 'storms_hist_circle.nc')

    def __str__(self):
        'Print paths'

        txt = ''
        for x in sorted(dir(self)):
            if x.startswith('p_'):
                txt+='\n{0:20} - {1}'.format(x, getattr(self,x))
        return txt[1:]
