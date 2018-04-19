#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import xarray as xr
import numpy as np

from lib.custom_dateutils import DateConverter_Mat2Py

def ReadMatfile(p_mfile):
    'Parse .mat file to nested python dictionaries'

    def RecursiveMatExplorer(mstruct_data):
        # Recursive function to extrat mat_struct nested contents

        if isinstance(mstruct_data, mat_struct):
            # mstruct_data is a matlab structure object, go deeper
            d_rc = {}
            for fn in mstruct_data._fieldnames:
                d_rc[fn] = RecursiveMatExplorer(getattr(mstruct_data, fn))
            return d_rc

        else:
            # mstruct_data is a numpy.ndarray, return value
            return mstruct_data

    # base matlab data will be in a dict
    mdata = sio.loadmat(p_mfile, squeeze_me=True, struct_as_record=False)
    mdata_keys = [x for x in mdata.keys() if x not in
                  ['__header__','__version__','__globals__']]

    # use recursive function
    dout = {}
    for k in mdata_keys:
        dout[k] = RecursiveMatExplorer(mdata[k])
    return dout

def ReadGowMat(p_mfile):
    'Read data from gow.mat file. Return xarray.Dataset'

    d_matf = ReadMatfile(p_mfile)

    # parse matlab datenum to datetime
    time = DateConverter_Mat2Py(d_matf['time'])

    # return xarray.Dataset
    return xr.Dataset(
        {
            'fp': (('time',), d_matf['fp']),
            'hs': (('time',), d_matf['hs']),
            't02': (('time',), d_matf['t02']),
            'dir': (('time',), d_matf['dir']),
            'spr': (('time',), d_matf['spr']),
            'hsCal': (('time',), d_matf['hsCal']),
        },
        coords = {
            'time': time
        },
        attrs = {
            'lat': d_matf['lat'],
            'lon': d_matf['lon'],
            'bat': d_matf['bat'],
            'forcing': d_matf['forcing'],
            'mesh': d_matf['mesh'],
        }
    )

def ReadCoastMat(p_mfile):
    '''
    Read coast polygons from Costa.mat file.
    Return list of NX2 np.array [x,y]
    '''

    d_matf = ReadMatfile(p_mfile)
    l_pol = []
    for ms in d_matf['costa']:
        l_pol.append(np.array([ms.x, ms.y]).T)
    return l_pol

