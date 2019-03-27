#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import numpy as np


def GetDivisors(x):
    l_div = []
    i = 1
    while i<x:
        if x%i == 0:
            l_div.append(i)
        i = i + 1
    return l_div

def GetUniqueRows(np_array):
    d = collections.OrderedDict()
    for a in np_array:
        t = tuple(a)
        if t in d:
            d[t] += 1
        else:
            d[t] = 1

    result = []
    for (key, value) in d.items():
        result.append(list(key) + [value])

    np_result = np.asarray(result)
    return np_result

