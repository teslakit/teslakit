#!/usr/bin/env python
# -*- coding: utf-8 -*-

def GetDivisors(x):
    l_div = []
    i = 1
    while i<x:
        if x%i == 0:
            l_div.append(i)
        i = i + 1
    return l_div
