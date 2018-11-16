#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.special import ndtri  # norm inv
from scipy.stats import t  # t student
import matplotlib.pyplot as plt

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

    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()

    # interpolate KDE CDF at x position (kde.support = x) 
    fint = interp1d(kde.support, kde.cdf)

    # plot CDF
    plotit = False
    if plotit:
        plt.plot(kde.support, kde.cdf,'k', label='')
        plt.plot(x, fint(x),'.r', label='fit points')
        plt.title('ksdensity CDF')
        plt.xlabel('x')
        plt.ylabel('CDF')
        plt.legend()
        plt.show()

    return fint(x)

def ksdensity_ICDF(x, p):
    '''
    Returns Inverse Kernel smoothing function at p points
    '''

    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()

    # interpolate KDE CDF to get support values 
    fint = interp1d(kde.cdf, kde.support)

    # ensure p inside kde.cdf
    p[p<np.min(kde.cdf)] = kde.cdf[0]
    p[p>np.max(kde.cdf)] = kde.cdf[-1]

    plotit = False
    if plotit:
        plt.plot(kde.cdf, kde.support, 'k', label='')
        plt.plot(p, fint(p), '.r', label='sim points')
        plt.title('ksdensity CDF')
        plt.xlabel('CDF')
        plt.ylabel('x')
        plt.legend()
        plt.show()

    return fint(p)

def copulafit(u, family='gaussian'):
    '''
    Fit copula to data.
    Returns correlation matrix and degrees of freedom for t student
    '''

    rhohat = None  # correlation matrix
    nuhat = None  # degrees of freedom (for t student) 

    if family=='gaussian':
        inv_n = ndtri(u)
        rhohat = np.corrcoef(inv_n.T)

    elif family=='t':
        raise ValueError("Not implemented")

        # TODO: no encaja con los datos. no funciona 
        x = np.linspace(np.min(u), np.max(u),100)
        inv_t = np.ndarray((len(x), u.shape[1]))

        for j in range(u.shape[1]):
            param = t.fit(u[:,j])
            t_pdf = t.pdf(x,loc=param[0],scale=param[1],df=param[2])
            inv_t[:,j] = t_pdf

        # TODO CORRELACION? NUHAT?
        rhohat = np.corrcoef(inv_n.T)
        nuhat = None

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return rhohat, nuhat

def copularnd(family, rhohat, n):
    '''
    Random vectors from a copula
    '''

    if family=='gaussian':
        mn = np.zeros(rhohat.shape[0])
        np_rmn = np.random.multivariate_normal(mn, rhohat, n)
        u = norm.cdf(np_rmn)

    elif family=='t':
        # TODO
        raise ValueError("Not implemented")

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return u

def runmean(X, m, modestr):
    '''
    parsed runmean function from original matlab codes.
    '''

    mm = 2*m+1

    if modestr == 'edge':
        xfirst = np.repeat(X[0], m)
        xlast = np.repeat(X[-1], m)
    elif modestr == 'zero':
        xfirst = np.zeros(m)
        xlast = np.zeros(m)
    elif modestr == 'mean':
        xfirst = np.repeat(np.mean(X), m)
        xlast = xfirst

    Y = np.concatenate(
        (np.zeros(1), xfirst, X, xlast)
    )
    Y = np.cumsum(Y)
    Y = np.divide(Y[mm:,]-Y[:-mm], mm)

    return Y

