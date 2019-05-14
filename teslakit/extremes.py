#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pip
import numpy as np
from scipy.stats import  gumbel_l, genextreme

def FitGEV_KMA_Frechet(bmus, n_clusters, var):
    '''
    Returns stationary GEV/Gumbel_L params for KMA bmus and varible series

    bmus - KMA bmus (time series of KMA centroids)
    n_clusters - number of KMA clusters
    var - time series of variable to fit to GEV/Gumbel_L

    returns np.array [n_clusters x 3 parameters (shape, loc, scale)]
    for gumbel distributions shape value will be ~0 (0.0000000001)
    '''

    param_GEV = np.empty((n_clusters, 3))
    for i in range(n_clusters):
        c = i+1
        pos = np.where((bmus==c))[0]

        if len(pos) == 0:
            param_GEV[i,:] = [np.nan, np.nan, np.nan]

        else:

            # get variable at cluster position
            var_c = var[pos]
            var_c = var_c[~np.isnan(var_c)]

            # fit to Gumbel_l and get negative loglikelihood
            loc_gl, scale_gl = gumbel_l.fit(-var_c)
            theta_gl = (0.0000000001, -1*loc_gl, scale_gl)
            nLogL_gl = genextreme.nnlf(theta_gl, var_c)

            # fit to GEV and get negative loglikelihood
            c = -0.1
            shape_gev, loc_gev, scale_gev = genextreme.fit(var_c, c)
            theta_gev = (shape_gev, loc_gev, scale_gev)
            nLogL_gev = genextreme.nnlf(theta_gev, var_c)

            # store negative shape
            theta_gev_fix = (-shape_gev, loc_gev, scale_gev)

            # apply significance test if Frechet
            if shape_gev < 0:
                if nLogL_gl - nLogL_gev >= 1.92:
                    param_GEV[i,:] = list(theta_gev_fix)
                else:
                    param_GEV[i,:] = list(theta_gl)
            else:
                param_GEV[i,:] = list(theta_gev_fix)

    return param_GEV

def Smooth_GEV_Shape(cenEOFs, param):
    '''
    TODO: Documentar
    '''
    n_cs = cenEOFs.shape[0]

    D = np.empty((n_cs, n_cs))
    sort_ord = np.empty((n_cs, n_cs), dtype=int)
    D_sorted = np.empty((n_cs, n_cs))

    param_c = np.empty(n_cs)

    for i in range(n_cs):
        for k in range(n_cs):
            D[i,k] = np.sqrt(np.sum(np.power(cenEOFs[i,:]-cenEOFs[k,:], 2)))
        D[i,i] = np.nan

        order = np.argsort(D[i,:])
        sort_ord[i,:] = order
        D_sorted[i,:] = D[i, order]

    for i in range(n_cs):
        denom = np.sum(
            [1/D_sorted[i,0], 1/D_sorted[i,1],
             1/D_sorted[i,2], 1/D_sorted[i,3]]
        )

        param_c[i] = 0.5 * np.sum(
            [
                param[i],
                param[sort_ord[i,0]] * np.divide(1/D_sorted[i,0], denom),
                param[sort_ord[i,1]] * np.divide(1/D_sorted[i,1], denom),
                param[sort_ord[i,2]] * np.divide(1/D_sorted[i,2], denom),
                param[sort_ord[i,3]] * np.divide(1/D_sorted[i,3], denom),
            ]
        )

    return param_c

def SampleGEV_KMA_Smooth(bmus, n_clusters, param_GEV, var):
    '''
    '''

    # TODO: en CLIMATE_EMULATOR. REFACTOR SI NECESARIO

    return param_GEV

def GEV_ACOV(theta, x):
    '''
    Returns asyntotyc variance matrix using Fisher Information matrix inverse

    theta = (shape, location, scale) GEV parameters
    '''

    # TODO: hace falta una funcion acov generica para varios parametros y
    # funcion L dada

    # TODO: simbolo shape correcto en todas partes?

    # gev shape, location and scale parameters
    k, u, s = theta

    # parameter differential
    pm = 0.0001
    dt_k = pm * k
    dt_u = pm * u
    dt_s = pm * s

    # second derivative evaluation 
    # dxx = (f(x+dt_x) - 2f(x) + f(x-dt_x)) / (dt_x**2)
    # dxy = (f(x,y) - f(x-dt_x,y) - f(x,y-dt_y) + f(x-dt_x, u-dt_y)) / (dt_x*dt_y)

    f = genextreme.nnlf  # GEV loglikelihood

    # TODO: EN ALGUN CASO NO RESUELVE Y DA INFINITO
    if np.isinf(f((k,u,s),x)):

        print ('GEV acov error: cant evaluate nLogL')
        return np.ones((3,3))*0.0001

    dkk = (f((k+dt_k,u,s),x) - 2*f(theta,x) + f((k-dt_k,u,s),x))/(dt_k**2)
    duu = (f((k,u+dt_u,s),x) - 2*f(theta,x) + f((k,u-dt_u,s),x))/(dt_u**2)
    dss = (f((k,u,s+dt_s),x) - 2*f(theta,x) + f((k,u,s-dt_s),x))/(dt_s**2)

    dku = (f(theta,x) - f((k-dt_k,u,s),x) - f((k,u-dt_u,s),x) + f((k-dt_k,u-dt_u,s),x))/(dt_k*dt_u)
    dks = (f(theta,x) - f((k-dt_k,u,s),x) - f((k,u,s-dt_s),x) + f((k-dt_k,u,s-dt_s),x))/(dt_k*dt_s)
    dus = (f(theta,x) - f((k,u-dt_u,s),x) - f((k,u,s-dt_s),x) + f((k,u-dt_u,s-dt_s),x))/(dt_s*dt_u)

    # Fisher Information matrix
    FI = np.array(
        [
            [dkk, dku, dks],
            [dku, duu, dus],
            [dks, dus, dss]
        ]
    )

    # asynptotic variance covariance matrix
    acov = np.linalg.inv(FI)

    return acov

