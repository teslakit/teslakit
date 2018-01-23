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
