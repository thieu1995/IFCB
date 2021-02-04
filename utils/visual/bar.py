#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:30, 14/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import arange, zeros, ones, ndarray
import platform


# Draw for different tasks, different models
def bar_chart_2d(list_g_bests: list, labels: list, names: list, list_color: list, filename: str, pathsave: list, exts: list):

    if len(list_g_bests) == 1:
        # list_g_best: in this case for the single model only
        y_pos = arange(len(names))
        plt.bar(y_pos, list_g_bests[0], align='center', alpha=0.85, color='red')
        plt.xticks(y_pos, names)
        plt.xlabel("Models")
        plt.ylabel(labels[0])

    for idx, ext in enumerate(exts):
        plt.savefig(pathsave[idx] + filename + ext)
    if platform.system() != "Linux":
        plt.show()
    plt.close()


def bar_chart_3d(list_g_bests: list, labels: list, names: list, list_color: list, filename: str, pathsave: list, exts: list, inside=True):
    fig = plt.figure()
    if inside:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = Axes3D(fig)
    if len(list_g_bests) == 1:
        # list_g_best: in this case for the single model only
        x3 = list(range(list_g_bests[0]))
        y3 = list(range(2*len(names)))
        z3 = zeros(len(list_g_bests[0]))
        dx = ones(len(list_g_bests[0]))
        dy = ones(len(list_g_bests[0]))
        dz = list_g_bests[0]

        ax.bar3d(x3, y3, z3, dx, dy, dz, color=list_color[0], label=names[0])
    else:
        for idx, g_bests in enumerate(list_g_bests):
            x3 = list(range(list_g_bests[idx]))
            y3 = list(range(2 * len(names)))
            z3 = zeros(len(list_g_bests[idx]))
            dx = ones(len(list_g_bests[idx]))
            dy = ones(len(list_g_bests[idx]))
            dz = list_g_bests[idx]

            ax.bar3d(x3, y3, z3, dx, dy, dz, color=list_color[idx], label=names[idx])

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.legend()
    for idx, ext in enumerate(exts):
        plt.savefig(pathsave[idx] + filename + ext, bbox_inches='tight')
    if platform.system() != "Linux":
        plt.show()
    plt.close()


def group_bar2d(groups:list, data:list, models:list, xy_labels:list, title:str, pathsave:list,
                filename:str, exts:list, auto_label=True):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    nsgaii = data[0]
    nsgaiii = data[1]
    moalo = data[2]
    mossa = data[3]

    x = np.arange(len(groups))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, nsgaii, width, label=models[0])
    rects2 = ax.bar(x + 1.0*width, nsgaiii, width, label=models[1])
    rects3 = ax.bar(x + 2.0*width, moalo, width, label=models[2])
    rects4 = ax.bar(x + 3.0*width, mossa, width, label=models[3])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(xy_labels[1])
    ax.set_xlabel(xy_labels[0])
    ax.set_title(title)
    ax.set_xticks(x+1.5*width)
    ax.set_xticklabels(groups)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    if auto_label:
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)

    for idx, ext in enumerate(exts):
        plt.savefig(pathsave[idx] + filename + ext, bbox_inches='tight')
    if platform.system() != "Linux":
        plt.show()
    plt.close()



def bar_2d_grouped():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ['100', '200', '300', '400', '500', '600', '700', '800', '900', '1000']
    nsgaii = [20, 34, 30, 35, 27, 20, 34, 30, 35, 27]
    nsgaiii = [25, 32, 34, 20, 25, 25, 32, 34, 20, 25]
    mossa = [18, 24, 46, 30, 28, 27, 41, 45, 29, 48]

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, nsgaii, width, label='NSGA-II')
    rects2 = ax.bar(x, nsgaiii, width, label='NSGA-III')
    rects3 = ax.bar(x + width, mossa, width, label='MO-SSA')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Service Latency')
    ax.set_xlabel("# Tasks")
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.show()


# bar_2d_grouped()