#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 11:17, 14/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from pathlib import Path
from copy import deepcopy
from sklearn.model_selection import ParameterGrid
from pandas import DataFrame
import pickle as pkl
from numpy import insert, array, concatenate

from config import Config, OptExp
from optimizer import BaseNSGA_II
from utils.io_util import load_tasks, load_nodes
from utils.visual.scatter import visualize_front_3d, visualize_front_2d, visualize_front_1d


def save_training_fitness_information(g_best_dict, number_tasks, name_mha, name_paras, results_folder_path):
    results_path = f'{results_folder_path}/optimize_process/{name_mha}/{name_paras}'
    Path(results_path).mkdir(parents=True, exist_ok=True)
    fitness_file_path = f'{results_path}/training_{number_tasks}_tasks.csv'
    fit_list = array([[0, 0, 0, 0]])
    for key, value in g_best_dict.items():
        value = insert(value, 0, key, axis=1)
        fit_list = concatenate((fit_list, value), axis=0)
    fitness_df = DataFrame(fit_list)
    fitness_df = fitness_df.iloc[1:]
    fitness_df.to_csv(fitness_file_path, index=False, header=["Epoch", "Power", "Latency", "Cost"])

    if Config.METRICS_NEED_MIN_OBJECTIVE_VALUES:
        fitness_df.drop(fitness_df[fitness_df['Epoch'] != key].index, inplace=True)
        fitness_df.insert(0, 'Name Paras', name_paras)
        fitness_df.insert(0, 'Name MHA', name_mha)
        fitness_df.insert(0, 'N Tasks', number_tasks)
        fitness_df.insert(0, 'Metric', Config.METRICS)
        fitness_df.to_csv(f'{Config.RESULTS_DATA}/summary.csv', mode='a', header=False)


def save_experiment_result(problem, solutions, g_best, name_mha, name_paras, results_folder_path):
    experiment_results_path = f'{results_folder_path}/experiment_results/{name_mha}/{name_paras}'
    Path(experiment_results_path).mkdir(parents=True, exist_ok=True)

    experiment_results_df = DataFrame(g_best)
    file_name = f'{experiment_results_path}/{len(problem["tasks"])}_tasks'
    experiment_results_df.index.name = "Solution"
    experiment_results_df.to_csv(f'{file_name}.csv', header=["Power", "Latency", "Cost"], index=True)

    schedule_object_save_path = open(f'{file_name}.pickle', 'wb')
    pkl.dump(solutions, schedule_object_save_path)
    schedule_object_save_path.close()


def save_visualization(problem, solution, name_mha, name_paras, results_folder_path):
    path_png = f'{results_folder_path}/visualization/{name_mha}/{name_paras}/png'
    path_pdf = f'{results_folder_path}/visualization/{name_mha}/{name_paras}/pdf'
    Path(path_png).mkdir(parents=True, exist_ok=True)
    Path(path_pdf).mkdir(parents=True, exist_ok=True)
    fn_3d = f'/{len(problem["tasks"])}_tasks-3d'
    fn_2d_PS = f'/{len(problem["tasks"])}_tasks-2d-PS'
    fn_2d_PM = f'/{len(problem["tasks"])}_tasks-2d-PM'
    fn_2d_SM = f'/{len(problem["tasks"])}_tasks-2d-SM'
    fn_1d_P = f'/{len(problem["tasks"])}_tasks-2d-P'
    fn_1d_S = f'/{len(problem["tasks"])}_tasks-2d-S'
    fn_1d_M = f'/{len(problem["tasks"])}_tasks-2d-M'

    visualize_front_3d([solution], ["Power Consumption", "Service Latency", "Monetary Cost"], ["NSGA-II"],
                       ["red"], ["o"], fn_3d, [path_png, path_pdf], [".png", ".pdf"], True)
    visualize_front_2d([solution[:, 0:2]], ["Power Consumption", "Service Latency"], ["NSGA-II"],
                       ["red"], ["o"], fn_2d_PS, [path_png, path_pdf], [".png", ".pdf"])
    visualize_front_2d([solution[:, [0, 2]]], ["Power Consumption", "Monetary Cost"], ["NSGA-II"],
                       ["red"], ["o"], fn_2d_PM, [path_png, path_pdf], [".png", ".pdf"])
    visualize_front_2d([solution[:, 1:3]], ["Service Latency", "Monetary Cost"], ["NSGA-II"],
                       ["red"], ["o"], fn_2d_SM, [path_png, path_pdf], [".png", ".pdf"])
    visualize_front_1d([solution[:, 0]], ["Power Consumption"], ["NSGA-II"],
                       ["red"], ["o"], fn_1d_P, [path_png, path_pdf], [".png", ".pdf"])
    visualize_front_1d([solution[:, 1]], ["Service Latency"], ["NSGA-II"],
                       ["red"], ["o"], fn_1d_S, [path_png, path_pdf], [".png", ".pdf"])
    visualize_front_1d([solution[:, 2]], ["Monetary Cost"], ["NSGA-II"],
                       ["red"], ["o"], fn_1d_M, [path_png, path_pdf], [".png", ".pdf"])


def inside_loop(my_model, n_trials, n_timebound):
    tasks = load_tasks(f'{Config.INPUT_DATA}/tasks_{my_model["n_tasks"]}.json')
    problem = deepcopy(my_model['problem'])
    problem["tasks"] = tasks
    problem["n_tasks"] = my_model["n_tasks"]
    problem["shape"] = [len(problem["clouds"]) + len(problem["fogs"]), my_model["n_tasks"]]

    # for paras in parameters_grid:
    if Config.MODE == "epoch":
        optimizer = BaseNSGA_II(problem, my_model["pop_size"], my_model["epoch"], my_model["func_eval"], my_model["lb"], my_model["ub"], paras=None)
        solutions, g_best, g_best_dict = optimizer.train()
    elif Config.MODE == "fe":
        optimizer = BaseNSGA_II(problem, my_model["pop_size"], my_model["epoch"], my_model["func_eval"], my_model["lb"], my_model["ub"], paras=None)
        solutions, g_best, g_best_dict = optimizer.train()

    if Config.TIME_BOUND_KEY:
        results_folder_path = f'{Config.RESULTS_DATA}_{n_timebound}s/{Config.METRICS}/{n_trials}'
    else:
        results_folder_path = f'{Config.RESULTS_DATA}_no_time_bound/{Config.METRICS}/{n_trials}'
    Path(results_folder_path).mkdir(parents=True, exist_ok=True)
    name_mha = 'nsgaii'
    name_paras = f'{my_model["epoch"]}_{my_model["pop_size"]}'

    save_training_fitness_information(g_best_dict, len(tasks), name_mha, name_paras, results_folder_path)
    save_experiment_result(problem, solutions, g_best, name_mha, name_paras, results_folder_path)
    save_visualization(problem, g_best, name_mha, name_paras, results_folder_path)


def optimize_schedule_with_nsgaii(my_model):
    print(f'Start running: {my_model["optimizer"]}')
    for n_trials in range(OptExp.N_TRIALS):
        if Config.TIME_BOUND_KEY:
            for n_timebound in OptExp.TIME_BOUND_VALUES:
                inside_loop(my_model, n_trials, n_timebound)
        else:
            inside_loop(my_model, n_trials, None)


if __name__ == "__main__":
    clouds, fogs, peers = load_nodes(f'{Config.INPUT_DATA}/nodes_2_8_5.json')
    problem = {
        "clouds": clouds,
        "fogs": fogs,
        "peers": peers,
        "n_clouds": len(clouds),
        "n_fogs": len(fogs),
        "n_peers": len(peers),
    }
    param_grid = {
        'n_tasks': [100],  # list(range(150, 201, 50))
        'pop_size': [50],  # [100]
        'epoch': [20],  # [200]
        'func_eval': [100000],
        'lb': [-1],
        'ub': [1],
        'optimizer': ['NSGA-II'],
        'problem': [problem]
    }
    for item in list(ParameterGrid(param_grid)):
        optimize_schedule_with_nsgaii(item)
