#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from scipy.stats import  gumbel_l, genextreme
from itertools import permutations


def ChromosomesProbabilities_KMA(bmus, n_clusters, vars_chrom):
    '''
    Calculate chromosomes probabilities for KMA bmus and variables at
    vars_chrom

    bmus - KMA bmus (time series of KMA centroids)
    n_clusters - number of KMA clusters
    vars_chrom - vars to get chromosomes probabilities, np.array (n x nvars)
                 nan data is considered a "0" chromosome, "1" otherwise
    '''

    def ChromMatrix(vs):
        'Return chromosome matrix for np.array vs (n x nvars)'

        chrom = np.empty((0,3), int)
        n_cols = vs.shape[1]
        b = np.zeros(n_cols)
        for c in range(n_cols):
            b[c] = 1
            for r in set(permutations(b.tolist())):
                chrom = np.row_stack([chrom, np.array(r)])

        return chrom

    # get chromosomes matrix
    chrom = ChromMatrix(vars_chrom)

    # calculate chromosomes probabilities
    probs = np.zeros((n_clusters, chrom.shape[0]))
    for i in range(n_clusters):
        c = i+1
        pos = np.where((bmus==c))[0]

        # get variables chromosomes at cluster
        var_c = vars_chrom[pos,:]
        var_c[~np.isnan(var_c)] = 1
        var_c[np.isnan(var_c)] = 0

        # count chromosomes
        ucs, ccs = np.unique(var_c, return_counts=True, axis=0)
        tcs = var_c.shape[0]

        # get probs of each chromosome
        for uc, cc in zip(ucs, ccs):

            # skip all empty chromosomes
            if ~uc.any(): continue

            pc = np.where(np.all(uc == chrom, axis=1))[0][0]
            probs[i, pc] = cc / tcs

    return chrom, probs


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
            if shape_gev < 0:  # TODO: signo alreves ??? hablar con Fer
                if nLogL_gl - nLogL_gev >= 1.92:
                    param_GEV[i,:] = list(theta_gev_fix)
                else:
                    param_GEV[i,:] = list(theta_gl)
            else:
                param_GEV[i,:] = list(theta_gev_fix)

    return param_GEV

def SampleGEV_KMA_Smooth(bmus, n_clusters, param_GEV, var):
    '''
    TODO: resolver/encontrar la matriz de informacion Fisher
    Generar un gev_params_sample (n_clustersx3) por simulacion
    respetando los param_GEV (n_clusterx3) input
    '''

    return param_GEV
    # TODO: NECESITO SACAR LA MATRIZ DE INFORMACION FISHER

    # gev parameters
    shape_gev = param_GEV[:,0]
    loc_gev = param_GEV[:,1]
    scale_gev = param_GEV[:,2]

    # location parameter
    index = np.ones((1, n_clusters))
    mu_strips = loc_gev - np.multiply(
        np.divide(scale_gev, shape_gev),
        1 - np.power(index, shape_gev)
    ).squeeze()

    psi_strips = np.multiply(
        scale_gev, np.power(index, shape_gev)
    )

    # Gumbel parameters
    pos_gumbel = np.where((shape_gev==0.0000000001))[0]
    loc_gumbel = -1*loc_gev[pos_gumbel]
    scale_gumbel = scale_gev[pos_gumbel]

    mu_strips[pos_gumbel] = loc_gumbel + np.multiply(
        scale_gumbel, np.log(index[:,pos_gumbel].squeeze())
    )

    # KMA bmus
    for i in range(n_clusters):
        c = i+1
        pos = np.where((bmus==c))[0]

        if len(pos) == 0:
            #param_SIM[i,:] = [np.nan, np.nan, np.nan]
            pass

        else:

            # get variable at cluster position
            var_c = var[pos]
            var_c = var_c[~np.isnan(var_c)]

            if i in pos_gumbel:
                pass

            else:
                # TODO ACOV?
                acov = genextreme.stats(
                    -1*shape_gev[i], loc_gev[i], scale_gev[i],
                    moments='sk'
                )
                pass

                #param_SIM = [shape_gev[i], mu_strips[i], psi_strips[i]]


    return None


