#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import urllib
import xarray as xr


def Download_HURRS(p_ncfile):
    '''
    Download HURRICANES (Allstorms) netcdf from NOAA

    ftp file name: Allstorms.ibtracs_all.v03rXX.nc.
    ftp wind velocity in knots: x1.82 km/h

    returns xarray.Dataset
    xds_HURRs:

        # TODO
        (time, ) VAR1
        (time, ) WVAR2
    '''

    # default parameters
    ftp_down = 'ftp://eclipse.ncdc.noaa.gov/pub/ibtracs/v03r10/wmo/netcdf/'
    fil_down = 'Allstorms.ibtracs_wmo.v03r10.nc.gz'

    p_down = op.dirname(p_ncfile)
    p_gz = op.join(p_down, '{0}'.format(fil_down))

    # download gz
    ftp_wmo = '{0}{1}'.format(ftp_down, fil_down)
    urllib.urlretrieve(ftp_wmo, p_gz)

    # TODO descomprimir gz

    # TODO: CONTINUAR CUANDO EL FTP VUELVA
    # TODO: no funciona mi ftp??

