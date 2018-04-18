#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import xarray as xr

def ReadSLP(p_db):
    'Read data from CFS SLP database: netCDF files'

    ncfiles = sorted(
        [op.join(p_db,f) for f in os.listdir(p_db) \
         if f.endswith('.nc') and 'gdas' in f]
    )

    # TODO: UN ARCHIVO PARA EL TEST
    ncfiles=ncfiles[0]

    # load entire dataset 
    print 'loading data from files...'
    if isinstance(ncfiles, str):
        print op.basename(ncfiles)
    else:
        print '\n'.join([op.basename(f) for f in ncfiles])

    xds_SLP = xr.open_mfdataset(ncfiles)

    # standarize dims
    xds_SLP = xds_SLP.rename(
        {'lat':'latitude',
         'lon':'longitude'}
    )

    return xds_SLP

