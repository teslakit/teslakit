#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def Plot_MJOphases(rmm1, rmm2, phase, p_export=None):
    'Plot MJO data separated by phase'

    # parameters for custom plot
    size_points = 0.2
    size_lines = 0.8
    l_colors_phase = np.array(
        [
            [1, 0, 0],
            [0.6602, 0.6602, 0.6602],
            [1.0, 0.4961, 0.3125],
            [0, 1, 0],
            [0.2539, 0.4102, 0.8789],
            [0, 1, 1],
            [1, 0.8398, 0],
            [0.2930, 0, 0.5078]
        ]
    )

    color_lines_1 = (0.4102, 0.4102, 0.4102)


    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(9,9))
    ax.scatter(rmm1, rmm2, c='b', s=size_points)

    # plot data by phases
    for i in range(1,9):
        ax.scatter(
            rmm1.where(phase==i),
            rmm2.where(phase==i),
            c=np.array([l_colors_phase[i-1]]),
            s=size_points)

    # plot sectors
    ax.plot([-4,4],[-4,4], color='k', linewidth=size_lines)
    ax.plot([-4,4],[4,-4], color='k', linewidth=size_lines)
    ax.plot([-4,4],[0,0],  color='k', linewidth=size_lines)
    ax.plot([0,0], [-4,4], color='k', linewidth=size_lines)

    # axis
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('RMM1')
    plt.ylabel('RMM2')
    ax.set_aspect('equal')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()

def Plot_MJOCategories(rmm1, rmm2, categ, p_export=None):
    'Plot MJO data separated by 25 categories'

    # parameters for custom plot
    size_lines = 0.8
    color_lines_1 = (0.4102, 0.4102, 0.4102)

    # category colors 
    l_colors_categ = np.array(
        [
            [0.527343750000000, 0.804687500000000, 0.979166666666667],
            [0, 0.746093750000000, 1],
            [0.253906250000000, 0.410156250000000, 0.878906250000000],
            [0, 0, 0.800781250000000],
            [0, 0, 0.542968750000000],
            [0.273437500000000, 0.507812500000000, 0.703125000000000],
            [0, 0.804687500000000, 0.816406250000000],
            [0.250000000000000, 0.875000000000000, 0.812500000000000],
            [0.500000000000000, 0, 0],
            [0.542968750000000, 0.269531250000000, 0.0742187500000000],
            [0.820312500000000, 0.410156250000000, 0.117187500000000],
            [1, 0.839843750000000, 0],
            [1, 0.644531250000000, 0],
            [1, 0.269531250000000, 0],
            [1, 0, 0],
            [0.695312500000000, 0.132812500000000, 0.132812500000000],
            [0.500000000000000, 0, 0.500000000000000],
            [0.597656250000000, 0.195312500000000, 0.796875000000000],
            [0.726562500000000, 0.332031250000000, 0.824218750000000],
            [1, 0, 1],
            [0.480468750000000, 0.406250000000000, 0.929687500000000],
            [0.539062500000000, 0.167968750000000, 0.882812500000000],
            [0.281250000000000, 0.238281250000000, 0.542968750000000],
            [0.292968750000000, 0, 0.507812500000000],
            [0.660156250000000, 0.660156250000000, 0.660156250000000],
        ]
    )

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(9,9))

    # plot sectors
    ax.plot([-4,4],[-4,4], color='k', linewidth=size_lines, zorder=9)
    ax.plot([-4,4],[4,-4], color='k', linewidth=size_lines, zorder=9)
    ax.plot([-4,4],[0,0],  color='k', linewidth=size_lines, zorder=9)
    ax.plot([0,0], [-4,4], color='k', linewidth=size_lines, zorder=9)

    # plot circles
    R = [1, 1.5, 2.5]

    for rr in R:
        ax.add_patch(
            patches.Circle(
                (0,0),
                rr,
                color='k',
                linewidth=size_lines,
                fill=False,
                zorder=9)
        )
    ax.add_patch(
        patches.Circle((0,0),R[0],fc='w',fill=True, zorder=10))

    # plot data by categories
    for i in range(1,25):
        if i>8:
            size_points = 0.2
        else:
            size_points = 1.7

        ax.scatter(
            rmm1.where(categ==i),
            rmm2.where(categ==i),
            c=[l_colors_categ[i-1]],
            s=size_points
        )

    ax.scatter(
        rmm1.where(categ==25),
        rmm2.where(categ==25),
        c=[l_colors_categ[i-1]],
        s=0.2,
        zorder=11
    )

    # axis
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('RMM1')
    plt.ylabel('RMM2')
    ax.set_aspect('equal')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()

