#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import numpy as np
from math import sqrt


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

def GetBestRowsCols(n):
    'try to square number n, used at gridspec plots'

    sqrt_n = sqrt(n)
    if sqrt_n.is_integer():
        n_r = int(sqrt_n)
        n_c = int(sqrt_n)
    else:
        l_div = GetDivisors(n)
        n_c = l_div[int(len(l_div)/2)]
        n_r = int(n/n_c)

    return n_r, n_c

