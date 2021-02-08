#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:05, 28/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

### Reading all results files to find True pareto-fronts (Reference Fronts)
from time import time
from pathlib import Path
from copy import deepcopy
from config import Config, OptExp, OptParas
from pandas import read_csv, DataFrame, to_numeric
from numpy import array, zeros, vstack, hstack, min, max, mean, std
from utils.io_util import load_tasks, load_nodes
from utils.metric_util import *
from utils.visual.scatter import visualize_front_3d


def inside_loop(my_model, n_trials, n_timebound, epoch, fe, end_paras):
    for pop_size in OptExp.POP_SIZE:
        if Config.TIME_BOUND_KEY:
            path_results = f'{Config.RESULTS_DATA}/{n_timebound}s/task_{my_model["problem"]["n_tasks"]}/{Config.METRICS}/{my_model["name"]}/{n_trials}'
        else:
            path_results = f'{Config.RESULTS_DATA}/no_time_bound/task_{my_model["problem"]["n_tasks"]}/{Config.METRICS}/{my_model["name"]}/{n_trials}'
        name_paras = f'{epoch}_{pop_size}_{end_paras}'
        file_name = f'{path_results}/experiment_results/{name_paras}-results.csv'
        df = read_csv(file_name, usecols=["Power", "Latency", "Cost"])
        return df.values

def getting_results_for_task(models):
    matrix_fit = zeros((1, 6))
    for n_task in OptExp.N_TASKS:
        for my_model in models:
            tasks = load_tasks(f'{Config.INPUT_DATA}/tasks_{n_task}.json')
            problem = deepcopy(my_model['problem'])
            problem["tasks"] = tasks
            problem["n_tasks"] = n_task
            problem["shape"] = [len(problem["clouds"]) + len(problem["fogs"]), n_task]
            my_model['problem'] = problem
            for n_trials in range(OptExp.N_TRIALS):
                if Config.TIME_BOUND_KEY:
                    for n_timebound in OptExp.TIME_BOUND_VALUES:
                        if Config.MODE == "epoch":
                            for epoch in OptExp.EPOCH:
                                end_paras = f"{epoch}"
                                df_matrix = inside_loop(my_model, n_trials, n_timebound, epoch, None, end_paras)
                                df_name = array([[n_task, my_model["name"], n_trials], ] * len(df_matrix))
                                matrix = hstack(df_name, df_matrix)
                                matrix_fit = vstack((matrix_fit, matrix))

                else:
                    if Config.MODE == "epoch":
                        for epoch in OptExp.EPOCH:
                            end_paras = f"{epoch}"
                            df_matrix = inside_loop(my_model, n_trials, None, epoch, None, end_paras)
                            df_name = array([[n_task, my_model["name"], n_trials], ] * len(df_matrix))
                            matrix = hstack((df_name, df_matrix))
                            matrix_fit = vstack((matrix_fit, matrix))
    return matrix_fit[1:]

starttime = time()
clouds, fogs, peers = load_nodes(f'{Config.INPUT_DATA}/nodes_2_8_5.json')
problem = {
    "clouds": clouds,
    "fogs": fogs,
    "peers": peers,
    "n_clouds": len(clouds),
    "n_fogs": len(fogs),
    "n_peers": len(peers),
}
models = [
    {"name": "NSGA-II", "class": "BaseNSGA_II", "param_grid": OptParas.NSGA_II, "problem": problem},
    {"name": "NSGA-III", "class": "BaseNSGA_III", "param_grid": OptParas.NSGA_III, "problem": problem},
    {"name": "MO-ALO", "class": "BaseMO_ALO", "param_grid": OptParas.MO_ALO, "problem": problem},
    {"name": "MO-SSA", "class": "BaseMO_SSA", "param_grid": OptParas.MO_SSA, "problem": problem},
]

## Load all results of all trials
matrix_results = getting_results_for_task(models)
# df_full = DataFrame(matrix_results, columns=["Task", "Model", "Trial", "Fit1", "Fit2", "Fit3"])


data = {'Task': matrix_results[:, 0],
        'Model': matrix_results[:, 1],
        'Trial': matrix_results[:, 2],
        'Fit1': matrix_results[:, 3],
        'Fit2': matrix_results[:, 4],
        'Fit3': matrix_results[:, 5],
        }
df_full = DataFrame(data)

df_full["Task"] = to_numeric(df_full["Task"])
df_full["Trial"] = to_numeric(df_full["Trial"])
df_full["Fit1"] = to_numeric(df_full["Fit1"])
df_full["Fit2"] = to_numeric(df_full["Fit2"])
df_full["Fit3"] = to_numeric(df_full["Fit3"])


for n_task in OptExp.N_TASKS:
    performance_results = []
    performance_results_mean = []

    ## Find matrix results for each problem
    df_task = df_full[df_full["Task"] == n_task]
    matrix_task = df_task[['Fit1', 'Fit2', 'Fit3']].values
    hyper_point = max(matrix_task, axis=0)

    ## Find non-dominated matrix for each problem
    reference_fronts = zeros((1, 3))
    dominated_list = find_dominates_list(matrix_task)
    for idx, value in enumerate(dominated_list):
        if value == 0:
            reference_fronts = vstack((reference_fronts, matrix_task[idx]))
    reference_fronts = reference_fronts[1:]

    ## For each model and each trial, calculate its performance metrics
    for model in models:
        er_list = zeros(OptExp.N_TRIALS)
        gd_list = zeros(OptExp.N_TRIALS)
        igd_list = zeros(OptExp.N_TRIALS)
        ste_list = zeros(OptExp.N_TRIALS)
        hv_list = zeros(OptExp.N_TRIALS)
        har_list = zeros(OptExp.N_TRIALS)

        for trial in range(OptExp.N_TRIALS):
            df_result = df_task[ (df_task["Model"] == model["name"]) & (df_task["Trial"] == trial) ]
            pareto_fronts = array(df_result.values[:, 3:], dtype=float)
            er = error_ratio(pareto_fronts, reference_fronts)
            gd = generational_distance(pareto_fronts, reference_fronts)
            igd = inverted_generational_distance(pareto_fronts, reference_fronts)
            ste = spacing_to_extent(pareto_fronts)
            hv = hyper_volume(pareto_fronts, reference_fronts, hyper_point, 100)
            har = hyper_area_ratio(pareto_fronts, reference_fronts, hyper_point, 100)
            performance_results.append([n_task, model["name"], trial, er, gd, igd, ste, hv, har])

            er_list[trial] = er
            gd_list[trial] = gd
            igd_list[trial] = igd
            ste_list[trial] = ste
            hv_list[trial] = hv
            har_list[trial] = har

        er_min, er_max, er_mean, er_std, er_cv = min(er_list), max(er_list), mean(er_list), std(er_list), std(er_list)/mean(er_list)
        gd_min, gd_max, gd_mean, gd_std, gd_cv = min(gd_list), max(gd_list), mean(gd_list), std(gd_list), std(gd_list)/mean(gd_list)
        igd_min, igd_max, igd_mean, igd_std, igd_cv = min(igd_list), max(igd_list), mean(igd_list), std(igd_list), std(igd_list)/mean(igd_list)
        ste_min, ste_max, ste_mean, ste_std, ste_cv = min(ste_list), max(ste_list), mean(ste_list), std(ste_list), std(ste_list)/mean(ste_list)
        hv_min, hv_max, hv_mean, hv_std, hv_cv = min(hv_list), max(hv_list), mean(hv_list), std(hv_list), std(hv_list) / mean(hv_list)
        har_min, har_max, har_mean, har_std, har_cv = min(har_list), max(har_list), mean(har_list), std(har_list), std(har_list) / mean(har_list)
        performance_results_mean.append([n_task, model["name"], er_min, er_max, er_mean, er_std, er_cv, gd_min, gd_max, gd_mean, gd_std, gd_cv,
                                         igd_min, igd_max, igd_mean, igd_std, igd_cv, ste_min, ste_max, ste_mean, ste_std, ste_cv,
                                         hv_min, hv_max, hv_mean, hv_std, hv_cv, har_min, har_max, har_mean, har_std, har_cv])

    filepath1 = f'{Config.RESULTS_DATA}/no_time_bound/task_{n_task}/{Config.METRICS}/metrics'
    Path(filepath1).mkdir(parents=True, exist_ok=True)
    df1 = DataFrame(performance_results, columns=["Task", "Model", "Trial", "ER", "GD", "IGD", "STE", "HV", "HAR"])
    df1.to_csv(f'{filepath1}/full_trials.csv', index=False)

    df2 = DataFrame(performance_results_mean, columns=["Task", "Model", "ER-MIN", "ER-MAX", "ER-MEAN", "ER-STD", "ER-CV",
                                                  "GD-MIN", "GD-MAX", "GD-MEAN", "GD-STD", "GD-CV",
                                                  "IGD-MIN", "IGD-MAX", "IGD-MEAN", "IGD-STD", "IGD-CV",
                                                  "STE-MIN", "STE-MAX", "STE-MEAN", "STE-STD", "STE-CV",
                                                       "HV-MIN", "HV-MAX", "HV-MEAN", "HV-STD", "HV-CV",
                                                       "HAR-MIN", "HAR-MAX", "HAR-MEAN", "HAR-STD", "HAR-CV"])
    df2.to_csv(f'{filepath1}/statistics.csv', index=False)


    ## Drawing some pareto-fronts founded. task --> trial ---> [modle1, model2, model3, ....]
    filepath3 = f'{Config.RESULTS_DATA}/no_time_bound/task_{n_task}/{Config.METRICS}/visual/'
    Path(filepath3).mkdir(parents=True, exist_ok=True)

    labels = ["Power Consumption (Wh)", "Service Latency (s)", "Monetary Cost ($)"]
    names = ["Reference Front"]
    list_color = [Config.VISUAL_FRONTS_COLORS[0]]
    list_marker = [Config.VISUAL_FRONTS_MARKERS[0]]
    for trial in range(OptExp.N_TRIALS):
        list_fronts = [reference_fronts, ]
        for idx, model in enumerate(models):
            df_result = df_task[ (df_task["Trial"] == trial) & (df_task["Model"] == model["name"]) ]
            list_fronts.append(df_result[['Fit1', 'Fit2', 'Fit3']].values)
            names.append(model["name"])
            list_color.append(Config.VISUAL_FRONTS_COLORS[idx+1])
            list_marker.append(Config.VISUAL_FRONTS_MARKERS[idx + 1])

        filename = f'pareto_fronts-{n_task}-{trial}'
        visualize_front_3d(list_fronts, labels, names, list_color, list_marker, filename, [filepath3, filepath3], inside=False)


print('That took: {} seconds'.format(time() - starttime))


