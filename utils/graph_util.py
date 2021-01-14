#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:36, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

from config import *


def visualize_front(points):
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D
    import random

    fig = pyplot.figure()
    ax = Axes3D(fig)

    # Generate the values
    x_vals = points[:, 0:1]
    y_vals = points[:, 1:2]
    z_vals = points[:, 2:3]

    ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    pyplot.show()


def visualize_mean_training_fitness():
    number_tasks = 150
    folder_name = 'trade-off_alpha_0.3333333333333333_beta_0.3333333333333333/optimize_process'
    method = ['pso', 'woa', 'ga', 'bla']
    colors = ['blue', 'green', 'orange']
    makers = ['o', '*', 'p', '.', 'P', 'v']
    number_tasks = 150
    for i in range(len(method)):
        results_training_fitness = []
        for j in range(0, 5, 1):
            if j == 0:
                file_results_experiment_time_path = f'{Config.RESULTS_DATA}/{folder_name}/{method[i]}/200_100_1/training_process_{number_tasks}.csv'
            else:
                file_results_experiment_time_path = f'{Config.RESULTS_DATA}_{j}/{folder_name}/{method[i]}/200_100_1/training_process_{number_tasks}.csv'
            _results_training_fitness = pd.read_csv(file_results_experiment_time_path, header=None).values

            results_training_fitness.append(_results_training_fitness)
        results_training_fitness = np.array(results_training_fitness)
        results_training_fitness = np.reshape(results_training_fitness, (results_training_fitness.shape[0], results_training_fitness.shape[1]))
        print(results_training_fitness.shape)
        min_0 = 0
        for k in range(results_training_fitness.shape[0]):
            min_0 += results_training_fitness[k][0] / 5
        print(min_0)
        print('---')
        min_1 = 0
        for k in range(results_training_fitness.shape[0]):
            min_1 += results_training_fitness[k][1] / 5
        print(min_1)
        mins = results_training_fitness.min(0)
        print(mins.shape)
        maxes = results_training_fitness.max(0)
        means = results_training_fitness.mean(0)
        std = results_training_fitness.std(0)
        slide = 10
        min_plot = []
        mean_plot = []
        max_plot = []
        std_plot = []
        for h in range(0, 200, 10):
            min_plot.append(mins[h])
            mean_plot.append(means[h])
            max_plot.append(maxes[h])
            std_plot.append(std[h])

        mean_plot.append(means[199])
        min_plot.append(mins[199])
        max_plot.append(maxes[199])

        min_plot = np.array(min_plot)
        max_plot = np.array(max_plot)
        mean_plot = np.array(mean_plot)
        std_plot = np.array(std_plot)
        print('method: ', method[i])
        print(max_plot)
        print(mean_plot)
        # plt.plot(means, label=method[i])
        # plt.errorbar(np.arange(20), mean_plot, fmt='ok', lw=3)
        a = np.arange(0, 200, 10)
        a = np.append(a, 199)
        # print(a)
        plt.xlabel('Iterations')
        plt.ylabel('Fitness value')
        plt.errorbar(a, mean_plot, [mean_plot - min_plot, max_plot - mean_plot],
                     fmt=makers[i], lw=2, xuplims=True, xlolims=True, label=method[i])
        plt.xlim(-3, 202, 1)
        plt.ylim(0.8, 0.96)
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(f'training_fitness_{number_tasks}.pdf')
    plt.savefig(f'training_fitness{number_tasks}.png')


def visulize_mean_trade_off_value():
    folder_name = 'trade-off_alpha_0.3333333333333333_beta_0.3333333333333333/optimize_process'
    method = ['pso', 'woa', 'ga', 'bla']
    maker_list = ['o', '*', 'v', 'p', 'P', '.']
    trade_off_value_bla = [0.7265048952778675, 0.8177141975992004, 0.838941727523262, 0.8642958838362544,
                           0.872965037248058, 0.8819704598977456, 0.8813933656484672, 0.8964888991735134,
                           0.8923133606268622, 0.9102349041797596]
    for i in range(len(method)):
        results_training_fitness = []
        for _number_tasks in range(50, 501, 50):
            _results_training_fitness = 0
            for j in range(0, 5, 1):
                if j == 0:
                    file_results_experiment_time_path = f'{Config.RESULTS_DATA}/{folder_name}/{method[i]}/200_100_1/training_process_{_number_tasks}.csv'
                else:
                    file_results_experiment_time_path = f'{Config.RESULTS_DATA}_{j}/{folder_name}/{method[i]}/200_100_1/training_process_{_number_tasks}.csv'

                if os.path.exists(file_results_experiment_time_path):
                    _results_training_fitness += pd.read_csv(file_results_experiment_time_path, header=None).values[-1][0] / 5
                else:
                    _results_training_fitness += trade_off_value_bla[int(_number_tasks / 50 - 1)] / 5
            results_training_fitness.append(_results_training_fitness)
        print(method[i], results_training_fitness)
        plt.plot(range(50, 501, 50), results_training_fitness, marker=maker_list[i], linestyle='--', label=method[i])
    trade_off_value_rr = [0.6528, 0.7617, 0.7873, 0.8009, 0.815, 0.835, 0.8446, 0.8591, 0.8501, 0.875]
    plt.plot(range(50, 501, 50), trade_off_value_rr, marker=maker_list[4], linestyle='--', label='RR')

    # plt.plot(range(50, 501, 50), trade_off_value_bla, marker=maker_list[5], linestyle='--', label='BLA')
    plt.xticks(range(50, 501, 50))
    plt.xlabel('Number of tasks')
    plt.ylabel('Fitness value')
    plt.ylim(0.6, 1)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('mean_trade_off_value.pdf')


def get_mean_contraint_value():
    mha = ['pso', 'woa']
    cost_rr = [211.83, 399.23, 590.76, 825.08, 1019.93, 1205.79, 1413.0, 1595.31, 1783.41, 1915.06]
    power_rr = [426.99, 886.44, 1396.33, 1995.91, 2617.13, 3120.02, 3743.02, 4192.97, 4945.6, 5466.25]
    latency_rr = [485.54, 964.41, 1524.12, 2157.8, 2709.13, 3180.14, 3832.06, 4300.4, 4668.95, 5302.26]

    for _number_task in range(50, 501, 50):
        task_str = f'{_number_task} & '
        power_str = ''
        latency_str = ''
        cost_str = ''

        for _mha in mha:
            mean_power = 0
            mean_latency = 0
            mean_cost = 0
            for i in range(5):
                if i == 0:
                    feature_experiment_results_path = f'{Config.RESULTS_DATA}/trade-off_alpha_0.3333333333333333_beta_0.3333333333333333/experiment_results/{_mha}/200_100_1'
                else:
                    feature_experiment_results_path = f'{Config.RESULTS_DATA}_{i}/trade-off_alpha_0.3333333333333333_beta_0.3333333333333333/experiment_results/{_mha}/200_100_1'

                results_file_path = f'{feature_experiment_results_path}/{_number_task}_tasks.csv'
                feature_value = pd.read_csv(results_file_path, header=None).values
                mean_power += (feature_value[0][0] / 5)
                mean_latency += (feature_value[1][0] / 5)
                mean_cost += (feature_value[2][0] / 5)
            mean_power = round(mean_power, 2)
            mean_latency = round(mean_latency, 2)
            mean_cost = round(mean_cost, 2)

            power_str += f' {mean_power} &'
            latency_str += f' {mean_latency} &'
            cost_str += f' {mean_cost} &'
        power_str += f' {power_rr[int(_number_task / 50 - 1)]} &'
        latency_str += f' {latency_rr[int(_number_task / 50 - 1)]} &'
        cost_str += f' {cost_rr[int(_number_task / 50 - 1)]} \\\\ \hline'
        task_str = task_str + power_str + latency_str + cost_str
        print(task_str)


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def visualize_mean_change_coeff_value(metrics, num_tasks):
    coeff_1 = [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]]

    coeff_2 = [[0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]

    coeff_3 = [[0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0],
               [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]]

    if metrics == 'Alpha':
        coeff = coeff_1
        x_axis = coeff[0]
    elif metrics == 'Beta':
        coeff = coeff_2
        x_axis = coeff[1]
    elif metrics == 'Alpha, Beta':
        coeff = coeff_3
        x_axis = coeff[0]
    # num_tasks = 200
    experiment_power = []
    experiment_latency = []
    experiment_cost = []
    for i in range(len(coeff[0])):
        mean_power = 0
        mean_latency = 0
        mean_cost = 0
        for j in range(5):
            if j == 0:
                result_path = f'{Config.RESULTS_DATA}/trade-off_alpha_{coeff[0][i]}_beta_{coeff[1][i]}/experiment_results/pso/200_100_1/{num_tasks}_tasks.csv'
            else:
                result_path = f'{Config.RESULTS_DATA}_{j}/trade-off_alpha_{coeff[0][i]}_beta_{coeff[1][i]}/experiment_results/pso/200_100_1/{num_tasks}_tasks.csv'
            value = pd.read_csv(result_path, header=None).values
            mean_power += value[0][0] / 5
            mean_latency += value[1][0] / 5
            mean_cost += value[2][0] / 5
        experiment_power.append(mean_power)
        experiment_latency.append(mean_latency)
        experiment_cost.append(mean_cost)
    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.2))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)

    p1, = host.plot(x_axis, experiment_power, "b-", marker='o', label="Power consumption")
    p2, = par1.plot(x_axis, experiment_latency, "r-", marker='P', label="Service latency")
    p3, = par2.plot(x_axis, experiment_cost, "g-", marker='*', label="Monetary cost")

    host.set_xlim(0.0, max(x_axis))
    host.set_ylim(math.floor(min(experiment_power)) - 50, math.ceil(max(experiment_power) + 50))
    par1.set_ylim(math.floor(min(experiment_latency)) - 50, math.ceil(max(experiment_latency) + 50))
    par2.set_ylim(math.floor(min(experiment_cost)) - 50, math.ceil(max(experiment_cost) + 50))
    print(f'=== {metrics}, {num_tasks} === ')
    print(round(experiment_power[0], 4), round(experiment_power[-1], 4))
    print(round(experiment_latency[0], 4), round(experiment_latency[-1], 4))
    print(round(experiment_cost[0], 4), round(experiment_cost[-1], 4))

    host.set_xlabel(metrics)
    host.set_ylabel("Power consumption")
    par1.set_ylabel("Service latency")
    par2.set_ylabel("Monetary cost")

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3]

    host.legend(lines, [l.get_label() for l in lines])

    # plt.show()
    plt.savefig(f'test_coeff_{num_tasks}_{metrics}.pdf')


# if __name__ == "__main__":
#     # visulize_mean_trade_off_value()
#     visualize_mean_training_fitness()

    # metrics = ['Alpha', 'Beta', 'Alpha, Beta']
    # num_tasks = [150]
    # for _metrics in metrics:
    #     for _num_tasks in num_tasks:
    #         visualize_mean_change_coeff_value(_metrics, _num_tasks)

    # get_mean_contraint_value()
