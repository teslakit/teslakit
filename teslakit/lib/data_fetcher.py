#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
from datetime import datetime


def Download_MJO(p_ncfile, init_year=None, log=False):
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
        ds_mjo = ds_mjo.loc[dict(time=slice(init_year, None))]

    # save at netcdf file
    ds_mjo.to_netcdf(p_ncfile,'w')

    if log:
        print '\nMJO historical data downloaded to \n{0}\nMJO time: {1} - {2}\n'.format(
            p_ncfile, ds_mjo.time.values[0],ds_mjo.time.values[-1])

    return ds_mjo

def Download_CSIRO(p_ncfile, lonq, latq, var_names):
    '''
    Download CSIRO data and stores it on netcdf format
    lonq, latq: longitude latitude query: single value or limits
    '''

    code = 'aus_4m'  # 'aus_4m', 'aus_10m', 'glob_24m', 'pac_4m', 'pac_10m'

    # long, lat query
    lonp1 = lonq[0]
    latp1 = latq[0]
    lonp2 = lonq[-1]
    latp2 = latq[-1]

    # parameters
    url_base = 'http://data-cbr.csiro.au/thredds/dodsC/catch_all/CMAR_CAWCR-Wave_archive/'
    url_1 = 'CAWCR_Wave_Hindcast_1979-2010/'
    url_2 = 'CAWCR_Wave_Hindcast_Jan_2011_-_May_2013/'
    url_3 = 'CAWCR_Wave_Hindcast_Jun_2013_-_Jul_2014/'

    # generate .nc url list
    l_ym = [
        '{0}{1:02d}'.format(x, y) for x in range(1979,2010+1) for y in range(1,13)]
    l_urls_1 = [
        '{0}{1}gridded/ww3.{2}.{3}.nc'.format(url_base, url_1, code, ym) for ym in l_ym]

    l_ym = [
        '{0}{1:02d}'.format(x, y) for x in range(2011,2013+1) for y in range(1,13)]
    l_ym = l_ym[:l_ym.index('201305')+1]
    l_urls_2 = [
        '{0}{1}gridded/ww3.{2}.{3}.nc'.format(url_base, url_2, code, ym) for ym in l_ym]

    l_ym = [
        '{0}{1:02d}'.format(x, y) for x in range(2013,2014+1) for y in range(1,13)]
    l_ym = l_ym[l_ym.index('201306'):l_ym.index('201407')+1]
    l_urls_3 = [
        '{0}{1}gridded/ww3.{2}.{3}.nc'.format(url_base, url_3, code, ym) for ym in l_ym]

    l_urls = l_urls_1 + l_urls_2 + l_urls_3


    # TODO: PODEMOS CORTAR LAS URLS PARA ACABAR DEV
    l_urls = l_urls[:2]


    # get coordinates from first file
    ff = xr.open_dataset(l_urls_1[0])
    idx1 = (np.abs(ff.longitude.values - lonp1)).argmin()
    idy1 = (np.abs(ff.latitude.values - latp1)).argmin()
    idx2 = (np.abs(ff.longitude.values - lonp2)).argmin()
    idy2 = (np.abs(ff.latitude.values - latp2)).argmin()

    # generate mem holder
    base = ff.isel(
        longitude=slice(idx1,idx2+1),
        latitude=slice(idy1,idy2+1))

    xds_out = xr.Dataset({},
        coords = {
            'time': [],
            'longitude': base.longitude.values,
            'latitude': base.latitude.values,
        }
    )

    # download data from files
    for u in l_urls:
        print u
        rxds = xr.open_dataset(u)
        cut = rxds.isel(
            longitude=slice(idx1,idx2+1),
            latitude=slice(idy1,idy2+1))
        xds_step = xr.Dataset({},
            coords = {
                'time': cut.time,
                'longitude': base.longitude.values,
                'latitude': base.latitude.values,
            }
        )
        for vn in var_names:
            xds_step[vn] = cut[vn]

        # merge
        xds_out = xds_out.merge(xds_step)


    # save to netcdf file
    xds_out.to_netcdf(p_ncfile,'w')

    return xr.open_dataset(p_ncfile)

