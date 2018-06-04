#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import os
import os.path as op
import netCDF4 as nc4
import xarray as xr
from datetime import datetime, timedelta


def Download_MJO(p_ncfile, init_year=None, log=False):
    '''
    Download MJO data and stores it on netcdf format
    init_year: optional, data before init_year will be discarded ('yyyy-mm-dd')
    log: optional, show log

    returns xarray.Dataset
    xds_MJO:
        (time, ) mjo
        (time, ) phase
        (time, ) rmm1
        (time, ) rmm2
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
        ds_mjo = ds_mjo.loc[dict(time=slice(init_year, None))]

    # save at netcdf file
    ds_mjo.to_netcdf(p_ncfile,'w')

    if log:
        print '\nMJO historical data downloaded to \n{0}\nMJO time: {1} - {2}\n'.format(
            p_ncfile, ds_mjo.time.values[0],ds_mjo.time.values[-1])

    return ds_mjo

