#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

def running_mean(x, N):
    'Same running mean as used by Dylan'

    nn = 2*N+1

    # case mean
    x = numpy.insert(x, 0, numpy.ones(N)*x[0])
    x = numpy.append(x, numpy.ones(N)*x[0])

    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
    return (cumsum[nn:] - cumsum[:-nn]) / float(nn)

def ClassificationKMA(PCA, num_clusters, num_reps, repres):
    'TODO: DOCUMENTAR'

    return None
