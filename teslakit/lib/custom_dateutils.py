#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import numpy as np

def datematlab2datetime(datenum_matlab):
    'Return python datetime for matlab datenum. Transform and adjust from matlab.'

    d = datetime.fromordinal(int(datenum_matlab)) + \
    timedelta(days=datenum_matlab % 1) - \
        timedelta(days=366) + timedelta(microseconds=0)

    return d

def datevec2datetime(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''

    return [datetime(d[0], d[1], d[2], d[3], d[4]) for d in d_vec]

def DateConverter_Mat2Py(datearray_matlab):
    'Parses matlab datenum array to python datetime list'

    return [datematlab2datetime(x) for x in datearray_matlab]

def xds2datetime(d64):
    'converts xr.Dataset np.datetime64[ns] into datetime'
    # TODO: MUY INEFICIENTE. Demasiado simple, eliminar si posible

    return datetime(d64.dt.year, d64.dt.month, d64.dt.day)

def npdt64todatetime(dt64):
    'converts np.datetime64[ns] into datetime'

    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(ts)

def xds_reindex_daily(xds_data,  dt_lim1=None, dt_lim2=None):
    '''
    Reindex xarray.Dataset to daily data between optional limits
    '''
    # TODO: demasiado simple, eliminar si posible

    if isinstance(xds_data.time.values[0], datetime):
        xds_dt1 = xds_data.time.values[0]
        xds_dt2 = xds_data.time.values[-1]
    else:
        # parse xds times to python datetime
        xds_dt1 = xds2datetime(xds_data.time[0])
        xds_dt2 = xds2datetime(xds_data.time[-1])

    # cut data at limits
    if dt_lim1:
        xds_dt1 = max(xds_dt1, dt_lim1)
    if dt_lim2:
        xds_dt2 = min(xds_dt2, dt_lim2)

    # number of days
    num_days = (xds_dt2-xds_dt1).days+1

    # reindex xarray.Dataset
    return xds_data.reindex(
        {'time': [xds_dt1 + timedelta(days=i) for i in range(num_days)]},
        method = 'pad',
    )

def xds_common_dates_daily(xds_list):
    '''
    returns daily datetime array between 2 xarray.Dataset common dates
    '''

    d1 = None
    d2 = None

    for xds_e in xds_list:

        if isinstance(xds_e.time.values[0], datetime):
            xds_e_dt1 = xds_e.time.values[0]
            xds_e_dt2 = xds_e.time.values[-1]
        else:
            # parse xds times to python datetime
            xds_e_dt1 = xds2datetime(xds_e.time[0])
            xds_e_dt2 = xds2datetime(xds_e.time[-1])

        if d1 == None:
            d1 = xds_e_dt1
            d2 = xds_e_dt2

        d1 = max(xds_e_dt1, d1)
        d2 = min(xds_e_dt2, d2)

    return [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

