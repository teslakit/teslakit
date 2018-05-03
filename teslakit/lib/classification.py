#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
from scipy import stats
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
import statsmodels.api as sm
from scipy.interpolate import interp1d
from datetime import datetime
from sklearn import linear_model


def ClassificationKMA(xds_PCA, num_clusters, repres):
    '''
    KMA Classification

    xds_PCA:
        (n_components, n_components) PCs
        (n_components, n_features) EOFs
        (n_components, ) variance
    num_clusters
    repres

    returns a xarray.Dataset containing KMA data
    '''
    # TODO: ACABAR COPULAS

    # PCA data
    variance = xds_PCA['variance']
    EOFs = xds_PCA['EOFs']
    PCs = xds_PCA['PCs']

    # APEV: the cummulative proportion of explained variance by ith PC
    APEV = np.cumsum(variance.values) / np.sum(variance.values)*100.0
    nterm = np.where(APEV <= repres*100)[0][-1]

    PCsub = PCs.values[:, :nterm+1]
    EOFsub = EOFs.values[:nterm+1, :]

    # KMEANS
    kma = KMeans(n_clusters=num_clusters, n_init=2000).fit(PCsub)

    # sort kmeans
    kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, num_clusters)
    bmus_corrected = np.zeros((len(kma.labels_),),)*np.nan
    for i in range(num_clusters):
        posc = np.where(kma.labels_==kma_order[i])
        bmus_corrected[posc] = i

    # adding some usefull data
    # TODO: km, x, SST_centers (igual entra en otra parte del codigo)

    # TODO: dates y bmus_corrected del codigo matlab?

    # Get bmus Persistences
    # TODO: ver como guardar esta info
    d_pers = Persistences(kma.labels_)

    # first 3 PCs
    PC1 = np.divide(PCsub[:,0], np.sqrt(variance.values[0]))
    PC2 = np.divide(PCsub[:,1], np.sqrt(variance.values[1]))
    PC3 = np.divide(PCsub[:,2], np.sqrt(variance.values[2]))

    # TODO ACABAR COPULAS
    # Generate copula for each WT
    for i in []:  #range(num_clusters):

        # getting copula number from plotting order
        num = kma_order[i]

        # find all the best match units equal
        ind = np.where(kma.labels_ == num)[:]

        # transfom data using kernel estimator
        cdf_PC1 = ksdensity_CDF(PC1[ind])
        cdf_PC2 = ksdensity_CDF(PC2[ind])
        cdf_PC3 = ksdensity_CDF(PC3[ind])
        U = np.column_stack((cdf_PC1.T, cdf_PC2.T, cdf_PC3.T))

        # TODO COPULAFIT. fit u to a student t copula. leer la web que compara

        # TODO COPULARND para crear USIMULADO

        # TODO: KS DENSITY ICDF PARA CREAR PC123_RND SIMULATODS

        # TODO: USAR NUM PARA GUARDAR LOS RESULTADOS

    print 'KMEANS Classification COMPLETE'
    return xr.Dataset(
        {
            'order': (('n_clusters'), kma_order),
            'bmus_corrected': (('n_pcacomp'), bmus_corrected.astype(int)),
            'cenEOFs': (('n_clusters', 'n_features'), kma.cluster_centers_),
            'bmus': (('n_pcacomp',), kma.labels_),
            'PCs': (('n_pcacomp','n_features'), PCsub),
            'centroids': (('n_clusters','n_pcafeat'),
                          np.dot(kma.cluster_centers_, EOFsub)),
            'PC1': (('n_pcacomp'), PC1),
            'PC2': (('n_pcacomp'), PC2),
            'PC3': (('n_pcacomp'), PC3),
        }
    )

def KMA_regression_guided(xds_PCA, num_clusters, repres):
    pass

def SimpleMultivariateRegressionModel(xds_PCA, xds_WAVES, name_vars):
    '''
    Regression model between daily predictor and predictand

    xds_PCA: predictor: SLP GRD PCAS
        (n_components, n_components) PCs
        (n_components, n_features) EOFs
        (n_components, ) variance

    xds_WAVES: predictand GOW waves data
        name_vars will be used as predictand (ex: ['hs','t02'])
        dim: time

    returns a xarray.Dataset
    '''

    repres = 0.951
    # TODO: NO HAY SEPARACION CALIBRACION / VALIDACION 

    # PREDICTOR: PCA data
    variance = xds_PCA['variance']
    EOFs = xds_PCA['EOFs']
    PCs = xds_PCA['PCs']

    # APEV: the cummulative proportion of explained variance by ith PC
    APEV = np.cumsum(variance.values) / np.sum(variance.values)*100.0
    nterm = np.where(APEV <= repres*100)[0][-1]

    PCsub = PCs.values[:, :nterm-1]
    EOFsub = EOFs.values[:nterm-1, :]

    PCsub_std = np.std(PCsub, axis=0)
    PCsub_norm = np.divide(PCsub, PCsub_std)

    X = PCsub_norm  # predictor

    # PREDICTAND: WAVES data
    wd = np.array([xds_WAVES[vn].values for vn in name_vars]).T
    wd_std = np.nanstd(wd, axis=0)
    wd_norm = np.divide(wd, wd_std)

    Y = wd_norm  # predictand

    # TODO separate validation / calibration data
    time_cal = xds_WAVES.time
    time_val = None
    X_cal = None
    Y_cal = None
    X_val = None
    Y_val = None


    # Adjust
    [n, d] = Y.shape
    X = np.concatenate((np.ones((n,1)), X), axis=1)
    clf = linear_model.LinearRegression(fit_intercept=True)
    Ymod = np.zeros((n,d))*np.nan
    for i in range(d):
        clf.fit(X, Y[:,i])
        beta = clf.coef_
        intercept = clf.intercept_
        Ymod[:,i] = np.ones((n,))*intercept
        for j in range(len(beta)):
            Ymod[:,i] = Ymod[:,i] + beta[j]*X[:,j]

    # de-scale
    Ym = np.multiply(Ymod, wd_std)

    return xr.Dataset(
        {
            'Ym': (('time', 'vars'), Ym),
            #'Ym_val': (('time, n_dimensions'), Ym),
        },
        {
            'time': time_cal,
            'vars': [vn for vn in name_vars],
        }
    )
