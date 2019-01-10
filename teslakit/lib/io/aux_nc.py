#!/usr/bin/env python
# -*- coding: utf-8 -*-

import netCDF4
from datetime import datetime, date


def StoreBugXdset(xds_data, p_ncfile):
    '''
    Stores xarray.Dataset to .nc file while avoiding bug with time data (>2262)
    '''

    # get metadata from xarray.Dataset
    dim_names = xds_data.dims.keys()
    var_names = xds_data.variables.keys()

    # Handle time data  (calendar format)
    calendar = 'standard'
    units = 'days since 1970-01-01 00:00:00'

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
            dv = root.createVariable(varname=vn,dimensions=(vn,), datatype='float32')

            if vn == 'time':  # time dimension values
                if isinstance(vals[0], date):
                    # parse datetime.date to datetime.datetime
                    vals = [datetime.combine(d,datetime.min.time()) for d in vals]

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


