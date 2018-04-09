#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

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
    'converts np.datetime64[ns] into datetime'

    return datetime(d64.dt.year, d64.dt.month, d64.dt.day)

def xds_reindex_daily(xds_data,  dt_lim1=None, dt_lim2=None):
    '''
    Reindex xarray.Dataset to daily data between optional limits
    '''

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

def xds_common_dates_daily(xds1, xds2):
    '''
    returns daily datetime array between 2 xarray.Dataset common dates
    '''

    # parse xds times to python datetime
    xds1_dt1 = xds2datetime(xds1.time[0])
    xds1_dt2 = xds2datetime(xds1.time[-1])
    xds2_dt1 = xds2datetime(xds2.time[0])
    xds2_dt2 = xds2datetime(xds2.time[-1])

    d1 = max(xds1_dt1, xds2_dt1)
    d2 = min(xds1_dt2, xds2_dt2)

    return [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

