#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d

def Persistences(series):
    'Return series persistences for each element'

    # output dict
    d_pers = {}
    for e in set(series):
        d_pers[e] = []

    # analize series
    e0 = None
    while series.any():

        # pol left
        e1 = series[0]
        series = np.delete(series, 0)

        if e1 != e0:
            d_pers[e1].append(1)
        else:
            d_pers[e1][-1]+=1

        # step forward
        e0 = e1

    return d_pers

def ksdensity_CDF(x):
    '''
    Kernel smoothing function estimate.
    Returns cumulative probability function at x.
    '''

    # TODO METER UN SWITCH EN ARGS para devolver kde.icdf interp

    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()

    fint = interp1d(kde.support, kde.cdf)
    return fint(x)

