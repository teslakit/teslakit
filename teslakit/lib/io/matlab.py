#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import h5py
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

def ReadEstelaMat(p_mfile):
    '''
    Read estela data from .mat file.
    Return xarray.Dataset
    '''

    threshold = 0

    with h5py.File(p_mfile, 'r') as mf:

        # mesh
        mesh_lon = mf['TP']['fullX_centred'][:]
        mesh_lat = mf['full']['Y'][:]
        coast = mf['coastcntr']

        mesh_lon[mesh_lon<0]=mesh_lon[mesh_lon<0] + 360
        longitude = mesh_lon[0,:]
        latitude = mesh_lat[:,0]

        # fields
        d_D = {}
        d_F = {}
        d_Fthreas = {}
        fds = mf['C']['traveldays_interp'].keys()
        for fd in fds:
            d_D[fd] = mf['C']['traveldays_interp'][fd][:]
            d_F[fd] = mf['C']['FEmedia_interp'][fd][:]
            d_Fthreas[fd] = d_F[fd] / np.nanmax(d_F[fd])

            # use threshold
            d_D[fd][d_Fthreas[fd]<threshold/100] = np.nan
            d_F[fd][d_Fthreas[fd]<threshold/100] = np.nan
            d_Fthreas[fd][d_Fthreas[fd]<threshold/100] = np.nan


    # return xarray.Dataset
    xdset = xr.Dataset(
        {
        },
        coords = {
            'longitude': longitude,
            'latitude': latitude,
        },
        attrs = {
        }
    )

    for k in d_D.keys():
        xdset.update({
            'D_{0}'.format(k):(('latitude','longitude'), d_D[fd]),
            'F_{0}'.format(k):(('latitude','longitude'), d_F[fd]),
            'Fthreas_{0}'.format(k):(('latitude','longitude'), d_Fthreas[fd])
        })

    return xdset

