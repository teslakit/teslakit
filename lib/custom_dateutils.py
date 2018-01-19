#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime as dt

def datevec2datetime(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''

    return [dt.datetime(d[0], d[1], d[2], d[3], d[4]) for d in d_vec]
