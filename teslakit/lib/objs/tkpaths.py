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

        # TODO: ORGANIZAR DB/SITE CON ANA

        # TODO: CREAR METHODO SETSITE CON UNA SUBRAMA DE PATHS DE SITE
        #INDEPENDIENTES

        # -----------------------------------------------
        # database paths
        self.p_data = p_data

        # storms
        self.p_db_storms = op.join(p_data, 'STORMS')
        self.p_db_nakajo_mats = op.join(self.p_db_storms, 'Nakajo_tracks')
        self.p_db_NOAA = op.join(self.p_db_storms, 'Allstorms.ibtracs_wmo.v03r10.nc')
        self.p_db_NOAA_fix = op.join(self.p_db_storms, 'Allstorms.ibtracs_wmo.v03r10_fix.nc')

        # waves
        self.p_db_waves = op.join(p_data, 'WAVES')

        # slp
        self.p_db_slp = op.join(p_data, 'CFS', 'prmsl')  # CFS SLP database

        # -----------------------------------------------
        # site paths
        self.p_st_test_estela = op.join(
            self.p_data, 'tests', 'tests_estela', 'Roi_Kwajalein')

        # SLP-Estela predictor
        self.p_st_SLP = op.join(self.p_st_test_estela, 'SLP.nc')
        self.p_st_estela_mat = op.join(self.p_st_test_estela, 'kwajalein_roi_obj.mat')
        self.p_st_PRED_SLP = op.join(self.p_st_test_estela, 'pred_SLP')

        self.p_st_gow_point = op.join(
            self.p_st_test_estela, 'gow2_062_ 9.50_167.25.mat')
        self.p_st_coast_mat = op.join(
            self.p_st_test_estela, 'Costa.mat')

        self.p_st_storms_hist_circle = op.join(self.p_db_storms, 'storms_hist_circle.nc')

        self.p_st_waves_part_p1 = op.join(
            self.p_db_waves, 'partitions','point1.mat')

    def __str__(self):
        'Print paths'

        txt = ''
        for x in sorted(dir(self)):
            if x.startswith('p_'):
                txt+='\n{0:20} - {1}'.format(x, getattr(self,x))
        return txt[1:]
