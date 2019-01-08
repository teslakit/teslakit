#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
from lib.custom_dateutils import npdt64todatetime

class Hydrograph(object):
    'Stores hydrograph data'

    def __init__(self):
        self.date_index = []
        self.dates = []

        self.indx_hydro = []
        self.numdays_hydro = []
        self.Hs_hydro = []
        self.Tp_hydro = []
        self.Dir_hydro = []

        self.TWL_max = []
        self.Hs_max = []

        self.MU = []
        self.TAU = []


def Calculate_Hydrographs(xds_KMA, xds_WAVES):
    '''
    Calculates intradaily hydrographs
    '''

    # solve intradaily bins
    bmus = xds_KMA.bmus.values[:]
    time_KMA = xds_KMA.time.values[:]

    d_bins = {}
    for i_wt in sorted(set(bmus)):

        # find WT indexes at KMA bmus
        indx = np.where((bmus == i_wt))[0]
        date_index = time_KMA[indx]

        # find hydrograms longer than 1 day
        diff_time = np.array([
            (b-a).astype('timedelta64[D]')/np.timedelta64(1,'D') \
            for a,b in zip(date_index, date_index[1:])])
        sep_hydro = np.where((diff_time > 1.0))[0]

        hydro_indx = []
        hydro_indx.append(indx[0:sep_hydro[0]+1])
        for m in range(len(sep_hydro)-2):
            hydro_indx.append(
                indx[sep_hydro[m]+1:sep_hydro[m+1]+1]
            )
        hydro_indx.append([indx[sep_hydro[len(sep_hydro)-1]-1]])
        num_days = [len(x) for x in hydro_indx]

        # initialize some output lists
        Hs_hydro = []
        Tp_hydro = []
        Dir_hydro = []
        TWL_max = []
        Hs_max = []
        MU = []
        TAU = []

        # work with storms <= 4 days
        ndays_storms = 4
        for h, nd in zip(hydro_indx, num_days):

            # initialize loop output
            hs_W = []
            tp_W = []
            dir_W = []
            twl_max_v = 0
            twl_max_t = 0
            hs_max_v = 0
            mu_v = 0

            if nd <= ndays_storms:

                # start and end dates for hydrograph
                p1 = npdt64todatetime(time_KMA[h[0]])
                p2 = npdt64todatetime(time_KMA[h[-1]]).replace(hour=23)

                # get waves conditions for hydrograph
                xds_W = xds_WAVES.sel(time=slice(p1, p2))
                hs_W = xds_W.hs.values[:]
                tp_W = xds_W.tp.values[:]
                dir_W = xds_W.dir.values[:]

                # calculate TWL max and normalize
                twl_temp = 0.1*(hs_W**0.5)*tp_W

                dt = 1.0/len(twl_temp)
                t_norm = np.arange(dt,1+dt,dt)

                i_twlmax = np.where(twl_temp == np.amax(twl_temp))[0]
                twl_max_v =  twl_temp[i_twlmax][0]
                twl_max_t =  t_norm[i_twlmax][0]

                # calculate MU
                mu_v = np.trapz(np.divide(twl_temp, twl_max_v), t_norm)

                # calculate max Hs and Tp
                hs_max_v = hs_W[np.where(hs_W == np.amax(hs_W))[0]][0]
                tp_max_v = tp_W[np.where(tp_W == np.amax(tp_W))[0]][0]

            # store values
            Hs_hydro.append(hs_W)
            Tp_hydro.append(tp_W)
            Dir_hydro.append(dir_W)
            TWL_max.append(twl_max_v)
            Hs_max.append(hs_max_v)
            MU.append(mu_v)
            TAU.append(twl_max_t)

        # bin key and hydrograph storage
        bin_k = 'bin{0:02d}'.format(i_wt)
        bin_hy = Hydrograph()

        bin_hy.date_index = indx
        bin_hy.dates = date_index
        bin_hy.indx_hydro = hydro_indx
        bin_hy.numdays_hydro = num_days
        bin_hy.Hs_hydro = Hs_hydro
        bin_hy.Tp_hydro = Tp_hydro
        bin_hy.Dir_hydro = Dir_hydro
        bin_hy.TWL_max = TWL_max
        bin_hy.Hs_max = Hs_max
        bin_hy.MU = MU
        bin_hy.TAU = TAU
        print MU
        import sys; sys.exit()

        d_bins[bin_k] = bin_hy
        print '{0} calculated'.format(bin_k)

    return d_bins

