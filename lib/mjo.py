#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
from datetime import datetime

def GetMJOCategories(rmm1, rmm2, phase):
    '''
    Divides MJO data in 25 categories.
    returns array with categories time series
    and corresponding rmm
    '''

    rmm = np.sqrt(rmm1**2+rmm2**2)

    categ = np.empty(rmm.shape)*np.nan

    for i in range(1,9):
        s = np.squeeze(np.where(phase == i))
        rmm_p = rmm[s]

        # TODO: VA LENTO. OPTIMIZAR
        # usar busqueda y remplazar por indices .where
        for j in s:
            if rmm[j] > 2.5:
                categ[j] = i
            elif rmm[j] > 1.5:
                categ[j] = i+8
            elif rmm[j] > 1:
                categ[j] = i+8*2
            elif rmm[j] <= 1:
                categ[j] = 25

    # get rmm_caot
    rmm_categ = {}
    for i in range(1,26):
        s = np.squeeze(np.where(categ == i))
        rmm_categ['cat_{0}'.format(i)] = np.column_stack((rmm1[s],rmm2[s]))

    return categ.astype(int), rmm_categ


def DownloadMJO(p_ncfile, init_year=None):
    '''
    Download MJO data and stores it on netcdf format
    init_year: optional, data before init_year will be discarded ('yyyy-mm-dd')
    '''

    # default parameter
    url_mjo = 'http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt'

    # download data and mount time array
    ddata = np.genfromtxt(
        url_mjo,
        skip_header=2,
        usecols=(0,1,2,3,4,5,6),
        dtype = None,
        names = ('year','month','day','RMM1','RMM2','phase','amplitude'),
    )

    # mount dattime array
    dtimes = [datetime(d['year'], d['month'], d['day']) for d in ddata]

    # parse data to xarray.Dataset
    ds_mjo = xr.Dataset(
        {
            'mjo'   :(('time',), ddata['amplitude']),
            'phase' :(('time',), ddata['phase']),
            'rmm1'  :(('time',), ddata['RMM1']),
            'rmm2'  :(('time',), ddata['RMM2']),
        },
        {'time' : dtimes}
    )

    # cut dataset if asked
    if init_year:
        ds_mjo =ds_mjo.loc[dict(time=slice(init_year, None))]

    # save at netcdf file
    ds_mjo.to_netcdf(p_ncfile,'w')

    #print '\nMJO historical data downloaded to \n{0}\nMJO time: {1} - {2}\n'.format(
    #    p_ncfile, ds_mjo.time.values[0],ds_mjo.time.values[-1])

    return ds_mjo

