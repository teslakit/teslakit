#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

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

def ClassificationKMA(PCA, num_clusters, num_reps, repres):
    'TODO: DOCUMENTAR'

    return None
