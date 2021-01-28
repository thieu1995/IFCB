#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 13:56, 28/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from pathlib import Path
from pandas import DataFrame
import pickle as pkl
from numpy import insert, array, concatenate

from config import Config
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

