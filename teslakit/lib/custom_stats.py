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

def running_mean(x, N, mode_str='mean'):
    '''
    computes a running mean (also known as moving average)
    on the elements of the vector X. It uses a window of 2*M+1 datapoints

    As always with filtering, the values of Y can be inaccurate at the
    edges. RUNMEAN(..., MODESTR) determines how the edges are treated. MODESTR can be
    one of the following strings:
      'edge'    : X is padded with first and last values along dimension
                  DIM (default)
      'zeros'   : X is padded with zeros
      'ones'    : X is padded with ones
      'mean'    : X is padded with the mean along dimension DIM

    X should not contains NaNs, yielding an all NaN result.
    '''

    # if nan in data, return nan array
    if np.isnan(x).any():
        return np.full(x.shape, np.nan)

    nn = 2*N+1

    if mode_str == 'zeros':
        x = np.insert(x, 0, np.zeros(N))
        x = np.append(x, np.zeros(N))

    elif mode_str == 'ones':
        x = np.insert(x, 0, np.ones(N))
        x = np.append(x, np.ones(N))

    elif mode_str == 'edge':
        x = np.insert(x, 0, np.ones(N)*x[0])
        x = np.append(x, np.ones(N)*x[-1])

    elif mode_str == 'mean':
        x = np.insert(x, 0, np.ones(N)*np.mean(x))
        x = np.append(x, np.ones(N)*np.mean(x))


    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[nn:] - cumsum[:-nn]) / float(nn)

def sort_cluster_gen_corr_end(centers, dimdim):
    '''
    SOMs alternative
    '''
    # TODO: DOCUMENTAR. BIEN PROGRAMADA PERO TESTEAR

    # get dimx, dimy
    dimy = np.floor(np.sqrt(dimdim)).astype(int)
    dimx = np.ceil(np.sqrt(dimdim)).astype(int)

    if not np.equal(dimx*dimy, dimdim):
        print 'ne'
        # TODO: RAISE ERROR
        pass

    dd = distance_matrix(centers, centers)
    qx = 0
    sc = np.random.permutation(dimdim).reshape(dimy, dimx)

    # get qx
    for i in range(dimy):
        for j in range(dimx):

            # row F-1
            if not i==0:
                qx += dd[sc[i-1,j], sc[i,j]]

                if not j==0:
                    qx += dd[sc[i-1,j-1], sc[i,j]]

                if not j+1==dimx:
                    qx += dd[sc[i-1,j+1], sc[i,j]]

            # row F
            if not j==0:
                qx += dd[sc[i,j-1], sc[i,j]]

            if not j+1==dimx:
                qx += dd[sc[i,j+1], sc[i,j]]

            # row F+1
            if not i+1==dimy:
                qx += dd[sc[i+1,j], sc[i,j]]

                if not j==0:
                    qx += dd[sc[i+1,j-1], sc[i,j]]

                if not j+1==dimx:
                    qx += dd[sc[i+1,j+1], sc[i,j]]

    # test permutations
    q=np.inf
    go_out = False
    for i in range(dimdim):
        if go_out:
            break

        go_out = True

        for j in range(dimdim):
            for k in range(dimdim):
                if len(np.unique([i,j,k]))==3:

                    u = sc.flatten('F')
                    u[i] = sc.flatten('F')[j]
                    u[j] = sc.flatten('F')[k]
                    u[k] = sc.flatten('F')[i]
                    u = u.reshape(dimy, dimx, order='F')

                    f=0
                    for ix in range(dimy):
                        for jx in range(dimx):

                            # row F-1
                            if not ix==0:
                                f += dd[u[ix-1,jx], u[ix,jx]]

                                if not jx==0:
                                    f += dd[u[ix-1,jx-1], u[ix,jx]]

                                if not jx+1==dimx:
                                    f += dd[u[ix-1,jx+1], u[ix,jx]]

                            # row F
                            if not jx==0:
                                f += dd[u[ix,jx-1], u[ix,jx]]

                            if not jx+1==dimx:
                                f += dd[u[ix,jx+1], u[ix,jx]]

                            # row F+1
                            if not ix+1==dimy:
                                f += dd[u[ix+1,jx], u[ix,jx]]

                                if not jx==0:
                                    f += dd[u[ix+1,jx-1], u[ix,jx]]

                                if not jx+1==dimx:
                                    f += dd[u[ix+1,jx+1], u[ix,jx]]

                    if f<=q:
                        q = f
                        sc = u

                        if q<=qx:
                            qx=q
                            go_out=False


    # print qx
    return sc.flatten('F')

def ClassificationKMA(xds_PCA, num_clusters, num_reps, repres):
    '''
    '''
    # TODO DOCUMENTAR
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
            'time': (('n_pcacomp'), [datetime(x,1,1) for x in xds_PCA._years]),
        }
    )


