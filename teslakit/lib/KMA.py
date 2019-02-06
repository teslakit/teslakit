#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
from scipy import stats
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from datetime import datetime
from sklearn import linear_model


def sort_cluster_gen_corr_end(centers, dimdim):
    '''
    SOMs alternative
    '''
    # TODO: DOCUMENTAR.

    # get dimx, dimy
    dimy = np.floor(np.sqrt(dimdim)).astype(int)
    dimx = np.ceil(np.sqrt(dimdim)).astype(int)

    if not np.equal(dimx*dimy, dimdim):
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

    return sc.flatten('F')

def KMA_simple(xds_PCA, num_clusters, repres=0.95):
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

    # PCA data
    variance = xds_PCA['variance']
    EOFs = xds_PCA['EOFs']
    PCs = xds_PCA['PCs']

    var_anom_std = xds_PCA.var_anom_std.values[:]
    var_anom_mean = xds_PCA.var_anom_mean.values[:]

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

    # groupsize
    _, group_size = np.unique(kma.labels_, return_counts=True)

    # groups
    d_groups = {}
    for k in range(num_clusters):
        d_groups['{0}'.format(k)] = np.where(kma.labels_==k)
    # TODO: STORE GROUPS WITHIN OUTPUT DATASET    

    # centroids
    centroids = np.dot(kma.cluster_centers_, EOFsub)

    # km, x and var_centers
    km = np.multiply(
        centroids,
        np.tile(var_anom_std, (num_clusters, 1))
    ) + np.tile(var_anom_mean, (num_clusters, 1))


    print('KMEANS classification COMPLETE.')
    return xr.Dataset(
        {
            'order': (('n_clusters'), kma_order),
            'bmus_corrected': (('n_pcacomp'), bmus_corrected.astype(int)),
            'cenEOFs': (('n_clusters', 'n_features'), kma.cluster_centers_),
            'bmus': (('n_pcacomp',), kma.labels_),
            'centroids': (('n_clusters','n_pcafeat'), centroids),
            'Km': (('n_clusters','n_pcafeat'), km),
            'group_size': (('n_clusters'), group_size),

            # PCA data
            'PCs': (('n_pcacomp','n_features'), PCsub),
            'variance': (('n_pcacomp',), variance),
        }
    )

def KMA_regression_guided(xds_PCA, xds_Yregres, num_clusters, repres=0.95, alpha=0.5):
    '''
    KMA Classification: regression guided

    xds_PCA:
        (n_components, n_components) PCs
        (n_components, n_features) EOFs
        (n_components, ) variance
    xds_Yregres:
        (time, vars) Ym
    num_clusters
    repres
    '''

    # PCA data
    variance = xds_PCA['variance']
    EOFs = xds_PCA['EOFs']
    PCs = xds_PCA['PCs']

    # Yregres data
    Y = xds_Yregres['Ym']

    # APEV: the cummulative proportion of explained variance by ith PC
    APEV = np.cumsum(variance.values) / np.sum(variance.values)*100.0
    nterm = np.where(APEV <= repres*100)[0][-1]

    nterm = nterm+1
    PCsub = PCs.values[:, :nterm]

    # append Yregres data to PCs
    data = np.concatenate((PCsub, Y), axis=1)
    data_std = np.std(data, axis=0)
    data_mean = np.mean(data, axis=0)

    # normalize but keep PCs weigth
    data_norm = np.ones(data.shape)*np.nan
    for i in range(PCsub.shape[1]):
        data_norm[:,i] = np.divide(data[:,i]-data_mean[i], data_std[0])
    for i in range(PCsub.shape[1],data.shape[1]):
        data_norm[:,i] = np.divide(data[:,i]-data_mean[i], data_std[i])

    # apply alpha (PCs - Yregress weight)
    data_a = np.concatenate(
        ((1-alpha)*data_norm[:,:nterm],
         alpha*data_norm[:,nterm:]),
        axis=1
    )

    # TODO: INTRODUCIR BUCLE QUE REPITA KMEANS HASTA UN MIN DE DATOS
    # KMEANS
    kma = KMeans(n_clusters=num_clusters, n_init=2000).fit(data_a)

    # groupsize
    _, group_size = np.unique(kma.labels_, return_counts=True)

    # groups
    d_groups = {}
    for k in range(num_clusters):
        d_groups['{0}'.format(k)] = np.where(kma.labels_==k)
    # TODO: STORE GROUPS WITHIN OUTPUT DATASET    

    # centroids
    centroids = np.zeros((num_clusters, data.shape[1]))
    for k in range(num_clusters):
        centroids[k,:] = np.mean(data[d_groups['{0}'.format(k)],:], axis=1)

    # sort kmeans
    kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, num_clusters)

    bmus_corrected = np.zeros((len(kma.labels_),),)*np.nan
    for i in range(num_clusters):
        posc = np.where(kma.labels_==kma_order[i])
        bmus_corrected[posc] = i

    # reorder centroids
    sorted_cenEOFs = kma.cluster_centers_[kma_order,:]
    sorted_centroids = centroids[kma_order,:]

    print('KMEANS regression-guided classification COMPLETE.')
    return xr.Dataset(
        {
            # KMA data
            'bmus': (('n_components',), kma.labels_),
            'cenEOFs': (('n_clusters', 'n_features'), kma.cluster_centers_),
            'centroids': (('n_clusters','n_features'), centroids),
            'group_size': (('n_clusters'), group_size),

            # sorted KMA data
            'sorted_order': (('n_clusters'), kma_order),
            'sorted_bmus': (('n_components'), bmus_corrected.astype(int)),
            'sorted_cenEOFs': (('n_clusters', 'n_features'), sorted_cenEOFs),
            'sorted_centroids': (('n_clusters','n_features'), sorted_centroids),

        },
        attrs = {
            'method': 'regression guided',
            'alpha': alpha,
        }
    )

def SimpleMultivariateRegressionModel(xds_PCA, xds_WAVES, name_vars):
    '''
    Regression model between daily predictor and predictand

    xds_PCA: predictor: SLP GRD PCAS
        (n_components, n_components) PCs
        (n_components, n_features) EOFs
        (n_components, ) variance

    xds_WAVES: predictand waves data
        name_vars will be used as predictand (ex: ['hs','t02'])
        dim: time

    returns a xarray.Dataset
    '''

    # 95% repres
    repres = 0.951

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

    # TODO: calculate errors

    return xr.Dataset(
        {
            'Ym': (('time', 'vars'), Ym),
        },
        {
            'time': xds_WAVES.time,
            'vars': [vn for vn in name_vars],
        }
    )

