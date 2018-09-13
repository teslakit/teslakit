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

        # site paths

    def __str__(self):
        'Print paths'

        txt = ''
        for x in dir(self):
            if x.startswith('p_'):
                txt+='\n{0:20} - {1}'.format(x, getattr(self,x))
        return txt[1:]
