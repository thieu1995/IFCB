#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:27, 14/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import arange
import platform

if platform.system() == "Linux":  # Linux: "Linux", Mac: "Darwin", Windows: "Windows"
    import matplotlib
    matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.


def visualize_front_3d(list_points: list, labels:list, names:list, list_color:list, list_marker:list,
                       filename:str, pathsave: list, exts: list, inside=True):
    fig = pyplot.figure()
    if inside:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = Axes3D(fig)

    if len(list_points) == 1:
        points = list_points[0]
        x_vals = points[:, 0:1]
        y_vals = points[:, 1:2]
        z_vals = points[:, 2:3]

        ax.scatter(x_vals, y_vals, z_vals, c=list_color[0], marker=list_marker[0], label=names[0])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    else:
        for idx, points in enumerate(list_points):
            xs = points[:, 0:1]
            ys = points[:, 1:2]
            zs = points[:, 2:3]
            ax.scatter(xs, ys, zs, c=list_color[idx], marker=list_marker[idx], label=names[idx])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.legend()
    for idx, ext in enumerate(exts):
        pyplot.savefig(pathsave[idx] + filename + ext, bbox_inches='tight')
    if platform.system() != "Linux":
        pyplot.show()
    pyplot.close()


def visualize_front_2d(list_points: list, labels: list, names:list, list_color: list, list_marker: list,
                       filename: str, pathsave:list, exts:list):
    if len(list_points) == 1:
        points = list_points[0]
        x_vals = points[:, 0:1]
        y_vals = points[:, 1:2]
        pyplot.scatter(x_vals, y_vals, c=list_color[0], marker=list_marker[0], label=names[0])
    else:
        for idx, points in enumerate(list_points):
            xs = points[:, 0:1]
            ys = points[:, 1:2]
            pyplot.scatter(xs, ys, c=list_color[idx], marker=list_marker[idx], label=names[idx])
    pyplot.xlabel(labels[0])
    pyplot.ylabel(labels[1])
    pyplot.legend(loc='upper left')
    for idx, ext in enumerate(exts):
        pyplot.savefig(pathsave[idx] + filename + ext, bbox_inches='tight')
    if platform.system() != "Linux":
        pyplot.show()
    pyplot.close()


def visualize_front_1d(list_points: list, labels: list, names: list, list_color: list, list_marker: list,
                       filename: str, pathsave:list, exts:list):
    if len(list_points) == 1:
        points = list_points[0]
        xs = arange(len(points))
        ys = points
        pyplot.plot(xs, ys, c=list_color[0], marker=list_marker[0], label=names[0])
    else:
        for idx, points in enumerate(list_points):
            xs = arange(len(points))
            ys = points
            pyplot.plot(xs, ys, c=list_color[idx], marker=list_marker[idx], label=names[idx])
    pyplot.xlabel("Solution")
    pyplot.ylabel(labels[0])
    pyplot.legend(loc='upper left')
    for idx, ext in enumerate(exts):
        pyplot.savefig(pathsave[idx] + filename + ext, bbox_inches='tight')
    if platform.system() != "Linux":
        pyplot.show()
    pyplot.close()

