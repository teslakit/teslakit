#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

# teslakit
from ..storms import GeoAzimuth

# import constants
from .config import _faspect, _fsize, _fdpi

def get_storm_linestyle(vel):
    if vel >= 60:
        return 'solid'
    elif vel >= 40:
        return 'dashed'
    elif vel >= 20:
        return 'dashdot'
    else:
        return 'dotted'

def get_storm_color(categ):

    dcs = {
        0 : 'green',
        1 : 'yellow',
        2 : 'orange',
        3 : 'red',
        4 : 'purple',
        5 : 'black',
    }

    return dcs[categ]


def axplot_Tracks_Circle(ax, lon_point, lat_point, r_point,
                         lon_storms, lat_storms, index_in, index_out,
                         categs, vel):
    'axes plot storm tracks inside radius circle'

    n_storms = len(categs)

    # plot point and  circle
    ax.plot(lon_point, lat_point, '.b', markersize=12, zorder=100)
    circle = plt.Circle(
        (lon_point, lat_point), r_point,
        edgecolor='k', fill=False)
    ax.add_patch(circle)

    # plot storms
    for lon_s, lat_s, i_i, i_o, v, c in zip(
        lon_storms, lat_storms, index_in, index_out, vel, categs):

        # get storm data
        lon_in = lon_s[i_i:i_o+1]
        lat_in = lat_s[i_i:i_o+1]

        # linestyle (velocity) and linecolor (category)
        ls = get_storm_linestyle(v)
        cs = get_storm_color(c)

        # plot storm and point at entry
        ax.plot(lon_in, lat_in, linestyle=ls, color=cs)
        ax.plot(lon_in[0], lat_in[0], '.', markersize=8, color=cs, zorder=99)

    # customize axes
    ax.axis('equal')
    ax.axis('off')
    ax.set_title(
        'Historical Tracks ({0} TCs inside R={1}º)'.format(n_storms, r_point),
        {'fontsize': 14, 'fontweight':'bold'}
    )

def axplot_Parametrized_Circle(ax, lon_point, lat_point, r_point,
                               delta_storms, gamma_storms, categs, vel):
    'axes plot storm parametrized tracks inside radius circle'

    n_storms = len(categs)

    # plot point and  circle
    ax.plot(lon_point, lat_point, '.b', markersize=12, zorder=100)
    circle = plt.Circle(
        (lon_point, lat_point), r_point,
        edgecolor='k', fill=False)
    ax.add_patch(circle)

    # TODO: def functions  track --> params / params --> track 

    # start-end points aux. opts.
    n = 1000
    ang = np.linspace(0, 2*np.pi, n)
    x_ang = r_point * np.cos(ang) + lat_point
    y_ang = r_point * np.sin(ang) + lon_point

    center_radius = np.asarray(
        [GeoAzimuth(lat_point, lon_point, x, y) for x, y in zip(x_ang, y_ang)])

    # plot storms
    for delta, gamma, v, c in zip(delta_storms, gamma_storms, vel, categs):

        # find most similar angle
        im = np.argmin(np.absolute(center_radius - delta))
        lon_entry = y_ang[im]
        lat_entry = x_ang[im]

        ent_radius = np.asarray(
            [GeoAzimuth(x, y, x_ang[im], y_ang[im]) for x, y in zip(
                x_ang, y_ang)])

        im2 = np.argmin(np.absolute(ent_radius - gamma))
        lon_exit = y_ang[im2]
        lat_exit = x_ang[im2]

        # linestyle (velocity) and linecolor (category)
        ls = get_storm_linestyle(v)
        cs = get_storm_color(c)

        # plot storm and point at entry
        ax.plot([lon_entry, lon_exit], [lat_entry, lat_exit], linestyle=ls, color=cs)
        ax.plot(lon_entry, lat_entry, '.', markersize=8, color=cs, zorder=99)

    # customize axes 
    ax.axis('equal')
    ax.axis('off')
    ax.set_title(
        'Parametrized Tracks',
        {'fontsize': 14, 'fontweight':'bold'}
    )

def axlegend_categ_vel(ax):
    'add custom legends (storm category) to axes'

    # category legend
    lls_cat = [
        Line2D([0], [0], color = 'black'),
        Line2D([0], [0], color = 'purple'),
        Line2D([0], [0], color = 'red'),
        Line2D([0], [0], color = 'orange'),
        Line2D([0], [0], color = 'yellow'),
        Line2D([0], [0], color = 'green'),
    ]
    leg_cat = Legend(
        ax, lls_cat, ['5','4','3','2','1','0'],
        title = 'Category', bbox_to_anchor = (1.01, 1), loc='upper left',
    )
    ax.add_artist(leg_cat)

    # velocity legend
    lls_vel = [
        Line2D([0], [0], color = 'black', ls = 'solid'),
        Line2D([0], [0], color = 'black', ls = 'dashed'),
        Line2D([0], [0], color = 'black', ls = 'dashdot'),
        Line2D([0], [0], color = 'black', ls = 'dotted'),
    ]
    leg_vel = Legend(
        ax, lls_vel, ['> 60 km/h', '> 40 km/h', '> 20 km/h', '< 20 km/h'],
        title = 'Velocity', bbox_to_anchor = (1.01, 0.7), loc= 'upper left',
    )
    ax.add_artist(leg_vel)


def Plot_TCs_TracksParams(TCs_tracks, TCs_params, p_export=None):
    'Plots storms tracks and storms parametrized'

    # custom wmo names
    nm_lon = 'lon_wmo'
    nm_lat = 'lat_wmo'

    # get data from historical tracks
    n_storms = TCs_tracks.storm.shape[0]
    lon_storms = TCs_tracks[nm_lon].values[:]
    lat_storms = TCs_tracks[nm_lat].values[:]

    # get data from parametrized tracks
    lon_point = TCs_params.point_lon
    lat_point = TCs_params.point_lat
    r_point = TCs_params.point_r

    index_in = TCs_params.index_in.values[:]
    index_out = TCs_params.index_out.values[:]
    categs_storms = TCs_params.category.values[:]
    vel_storms = TCs_params.velocity_mean.values[:]
    gamma_storms = TCs_params.gamma.values[:]
    delta_storms = TCs_params.delta.values[:]

    # figure
    fig, axs = plt.subplots(ncols=2, figsize=(_faspect*_fsize, _fsize))

    # storm tracks
    axplot_Tracks_Circle(
        axs[0], lon_point, lat_point, r_point,
        lon_storms, lat_storms, index_in, index_out,
        categs_storms, vel_storms
    )

    # storms  parametrized
    axplot_Parametrized_Circle(
        axs[1], lon_point, lat_point, r_point,
        delta_storms, gamma_storms, categs_storms, vel_storms
    )

    # small axis lims fix
    axs[0].set_xlim(axs[1].get_xlim())
    axs[0].set_ylim(axs[1].get_ylim())

    # add custom legends
    axlegend_categ_vel(axs[1])

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

