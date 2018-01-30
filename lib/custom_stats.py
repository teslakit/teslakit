#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from sklearn.cluster import KMeans

def running_mean(x, N, mode_str='mean'):
    'Same running mean as used by Dylan'
    # TODO: INTRODUCIR EL SWITCH EDGE, ZERO, MEAN (usar var filler)

    # if nan in data, return nan array
    if numpy.isnan(x).any():
        return numpy.full(x.shape, numpy.nan)

    nn = 2*N+1

    # case zeros
    #x = numpy.insert(x, 0, numpy.zeros(N)*x[0])
    #x = numpy.append(x, numpy.zeros(N)*x[0])

    # case ones
    #x = numpy.insert(x, 0, numpy.ones(N)*x[0])
    #x = numpy.append(x, numpy.ones(N)*x[0])

    # case edge TODO
    #x = numpy.insert(x, 0
    #x = numpy.append(x, 

    # case mean (default)
    x = numpy.insert(x, 0, numpy.ones(N)*numpy.mean(x))
    x = numpy.append(x, numpy.ones(N)*numpy.mean(x))

    cumsum = numpy.cumsum(numpy.insert(x, 0, 0))
    return (cumsum[nn:] - cumsum[:-nn]) / float(nn)


def ClassificationKMA(d_PCA, num_clusters, num_reps, repres):
    'TODO: DOCUMENTAR'

    # PCA data
    variance = d_PCA['variance']
    EOFs = d_PCA['EOFs']
    PCs = d_PCA['PCs']

    # APEV: the cummulative proportion of explained variance by ith PC
    APEV = numpy.cumsum(variance.values) / numpy.sum(variance.values)*100.0
    nterm = numpy.where(APEV <= repres*100)[0][-1]

    PCsub = PCs[:, :nterm+1]
    EOFsub = EOFs[:nterm+1, :]

    # KMEANS
    kma = KMeans(n_clusters=num_clusters, n_init=2000).fit(PCsub)

    print 'KMEANS Classification COMPLETE'
    print kma

    # TODO: DEVOLVER XARRAY?
    return {
        'Nterm': nterm,
        'PCs': PCsub,
        'cenEOFs': kma.cluster_centers_,
        'bmus': kma.labels_,
        'centroids': numpy.dot(kma.cluster_centers_, EOFsub)
    }
    # TODO: dates y bmus_corrected del codigo matlab?

