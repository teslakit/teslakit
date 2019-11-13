#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
from datetime import datetime, date
import os
import os.path as op

# pip
import netCDF4
import numpy as np

# tk
from ..util.time_operations import npdt64todatetime as n2d


def StoreBugXdset(xds_data, p_ncfile):
    '''
    Stores xarray.Dataset to .nc file while avoiding bug with time data (>2262)
    '''
    # TODO: update pip libs and check if this is needed

    # get metadata from xarray.Dataset
    dim_names = xds_data.dims.keys()
    var_names = xds_data.variables.keys()

    # Handle time data  (calendar format)
    calendar = 'standard'
    units = 'days since 1970-01-01 00:00:00'

    # remove previous file
    if op.isfile(p_ncfile):
        os.remove(p_ncfile)

    # Use netCDF4 lib 
    root = netCDF4.Dataset(p_ncfile, 'w', format='NETCDF4')

    # Handle dimensions
    for dn in dim_names:
        vals = xds_data[dn].values[:]
        root.createDimension(dn, len(vals))

    # handle variables
    for vn in var_names:
        vals = xds_data[vn].values[:]

        # dimensions values
        if vn in dim_names:
            dv = root.createVariable(varname=vn, dimensions=(vn,), datatype='float32')

            if vn == 'time':  # time dimension values
                if isinstance(vals[0], date):
                    # parse datetime.date to datetime.datetime
                    vals = [datetime.combine(d, datetime.min.time()) for d in vals]
                    # TODO: needed? rounds to day...
                    #vals = vals

                elif isinstance(vals[0], np.datetime64):
                    # parse numpy.datetime64 to datetime.datetime
                    vals = [n2d(d) for d in vals]

                dv[:] = netCDF4.date2num(vals, units=units, calendar=calendar)
                dv.units = units
            else:
                dv[:] = vals

        # variables values
        else:
            vdims = xds_data[vn].dims
            vv = root.createVariable(varname=vn,dimensions=vdims, datatype='float32')
            vv[:] = vals

    # close file
    root.close()


