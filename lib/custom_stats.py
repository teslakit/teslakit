#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
import xarray as xr

def running_mean(x, N, mode_str='mean'):
    'Same running mean as used by Dylan'
    # TODO: INTRODUCIR EL SWITCH EDGE, ZERO, MEAN (usar var filler)

    # if nan in data, return nan array
    if np.isnan(x).any():
        return np.full(x.shape, np.nan)

    nn = 2*N+1

    # case zeros
    #x = np.insert(x, 0, np.zeros(N)*x[0])
    #x = np.append(x, np.zeros(N)*x[0])

    # case ones
    #x = np.insert(x, 0, np.ones(N)*x[0])
    #x = np.append(x, np.ones(N)*x[0])

    # case edge TODO
    #x = np.insert(x, 0
    #x = np.append(x, 

    # case mean (default)
    x = np.insert(x, 0, np.ones(N)*np.mean(x))
    x = np.append(x, np.ones(N)*np.mean(x))

    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[nn:] - cumsum[:-nn]) / float(nn)


def sort_cluster_gen_corr_end(centers, dimdim):
    'SOMs alternative'
    # TODO: DOCUMENTAR


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

    for i in range(dimy):
        for j in range(dimx):

            # row F-1
            if not i==0:
                pass

            # row F
            if not j==0:
                pass

            if not j+1==dimx:
                qx += dd[sc[i,j+1], sc[i,j]]

    import sys; sys.exit()


def ClassificationKMA(xds_PCA, num_clusters, num_reps, repres):
    ''
    # TODO DOCUMENTAR

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

    # TODO generar/almacenar el bmus corregido
    sort_data = sort_cluster_gen_corr_end(kma.cluster_centers_, num_clusters)

    # TODO: dates y bmus_corrected del codigo matlab?

    print 'KMEANS Classification COMPLETE'
    print kma

    return xr.Dataset(
        {
            'cenEOFs': (('n_clusters', 'n_features'), kma.cluster_centers_),
            'bmus': (('n_pcacomp',), kma.labels_),
            'PCs': (('n_pcacomp','n_features'), PCsub),
            'centroids': (('n_clusters','n_pcafeat'),
                          np.dot(kma.cluster_centers_, EOFsub)),
        }
    )


