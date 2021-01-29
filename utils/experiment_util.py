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


def save_experiment_results_multi(solutions, g_best, g_best_dict, name_paras, path_results, save_training=True):
    ## Save results
    path_results = f'{path_results}/experiment_results'
    Path(path_results).mkdir(parents=True, exist_ok=True)

    ## Save fitness
    df1 = DataFrame(g_best)
    file_name = f'{path_results}/{name_paras}'
    df1.index.name = "Solution"
    df1.to_csv(f'{file_name}-results.csv', header=["Power", "Latency", "Cost"], index=True)

    ## Save solution
    schedule_object_save_path = open(f'{file_name}-solution.pkl', 'wb')
    pkl.dump(solutions, schedule_object_save_path)
    schedule_object_save_path.close()

    ## Save training process
    if save_training:
        fit_list = array([[0, 0, 0, 0]])
        for key, value in g_best_dict.items():
            value = insert(value, 0, key, axis=1)
            fit_list = concatenate((fit_list, value), axis=0)
        fitness_df = DataFrame(fit_list)
        fitness_df = fitness_df.iloc[1:]
        fitness_df.to_csv(f'{file_name}-training.csv', index=False, header=["Epoch", "Power", "Latency", "Cost"])


def save_visualization_results_multi(solution, name_model, name_paras, path_results):
    from utils.visual.scatter import visualize_front_3d, visualize_front_2d, visualize_front_1d

    path_png = f'{path_results}/visualization/png'
    path_pdf = f'{path_results}/visualization/pdf'
    Path(path_png).mkdir(parents=True, exist_ok=True)
    Path(path_pdf).mkdir(parents=True, exist_ok=True)
    file_name = f'{path_results}/{name_paras}'
    fn_3d = f'/{file_name}-3d'
    fn_2d_PS = f'/{file_name}-2d-PS'
    fn_2d_PM = f'/{file_name}-2d-PM'
    fn_2d_SM = f'/{file_name}-2d-SM'
    fn_1d_P = f'/{file_name}-2d-P'
    fn_1d_S = f'/{file_name}-2d-S'
    fn_1d_M = f'/{file_name}-2d-M'

    visualize_front_3d([solution], Config.OBJ_NAME_1, [name_model],["red"], ["o"], fn_3d, [path_png, path_pdf], [".png", ".pdf"], True)
    visualize_front_2d([solution[:, 0:2]], Config.OBJ_NAME_2, [name_model], ["red"], ["o"], fn_2d_PS, [path_png, path_pdf], [".png", ".pdf"])
    visualize_front_2d([solution[:, [0, 2]]], Config.OBJ_NAME_3, [name_model], ["red"], ["o"], fn_2d_PM, [path_png, path_pdf], [".png", ".pdf"])
    visualize_front_2d([solution[:, 1:3]], Config.OBJ_NAME_4, [name_model], ["red"], ["o"], fn_2d_SM, [path_png, path_pdf], [".png", ".pdf"])
    visualize_front_1d([solution[:, 0]], Config.OBJ_NAME_5, [name_model], ["red"], ["o"], fn_1d_P, [path_png, path_pdf], [".png", ".pdf"])
    visualize_front_1d([solution[:, 1]], Config.OBJ_NAME_6, [name_model], ["red"], ["o"], fn_1d_S, [path_png, path_pdf], [".png", ".pdf"])
    visualize_front_1d([solution[:, 2]], Config.OBJ_NAME_7, [name_model], ["red"], ["o"], fn_1d_M, [path_png, path_pdf], [".png", ".pdf"])


def save_experiment_results_single(problem, solution, list_fitness, name_paras, time_total, path_results, save_training=True):
    from model.fitness import Fitness
    from utils.schedule_util import matrix_to_schedule

    ## Saving fitness
    path_results = f'{path_results}/experiment_results'
    Path(path_results).mkdir(parents=True, exist_ok=True)

    fit_obj = Fitness(problem)
    schedule = matrix_to_schedule(problem, solution)
    power = fit_obj.calc_power_consumption(schedule)
    latency = fit_obj.calc_latency(schedule)
    cost = fit_obj.calc_cost(schedule)
    fitness = fit_obj.fitness(schedule)
    file_name = f'{path_results}/{name_paras}'
    experiment_results = array([[power, latency, cost, fitness, time_total]])
    df1 = DataFrame(experiment_results)
    df1.index.name = "Solution"
    df1.to_csv(f'{file_name}-results.csv', header=["Power", "Latency", "Cost", "Fitness", "Time"], index=True)

    ## Saving model
    schedule_object_save_path = open(f'{file_name}-solution.pkl', 'wb')
    pkl.dump(schedule, schedule_object_save_path)
    schedule_object_save_path.close()

    ## Saving training process
    if save_training:
        fitness_df = DataFrame(list_fitness)
        fitness_df.index.name = "epoch"
        fitness_df.to_csv(f'{file_name}-training.csv', index=True, header=["fitness"])


def save_visualization_results_single(problem, solution, best_fit, name_model, name_paras, path_results):
    from model.fitness import Fitness
    from utils.schedule_util import matrix_to_schedule
    from utils.visual.bar import bar_chart_2d

    path_png = f'{path_results}/visualization/png'
    path_pdf = f'{path_results}/visualization/pdf'
    Path(path_png).mkdir(parents=True, exist_ok=True)
    Path(path_pdf).mkdir(parents=True, exist_ok=True)
    file_name = f'{path_results}/{name_paras}'

    fit_obj = Fitness(problem)
    schedule = matrix_to_schedule(problem, solution)
    power = fit_obj.calc_power_consumption(schedule)
    latency = fit_obj.calc_latency(schedule)
    cost = fit_obj.calc_cost(schedule)

    fn_2d_power = f'/{file_name}-2d-power'
    fn_2d_latency = f'/{file_name}-2d-latency'
    fn_2d_cost = f'/{file_name}-2d-cost'
    fn_2d_fit = f'/{file_name}-2d-fit'

    bar_chart_2d([best_fit], [f'fitness: {Config.METRICS}'], [name_model], ["red"], fn_2d_fit, [path_png, path_pdf], [".png", ".pdf"])
    bar_chart_2d([power], [f'Power Consumption'], [name_model], ["red"], fn_2d_power, [path_png, path_pdf], [".png", ".pdf"])
    bar_chart_2d([latency], [f'Service Latency'], [name_model], ["red"], fn_2d_latency, [path_png, path_pdf], [".png", ".pdf"])
    bar_chart_2d([cost], [f'Monetary Cost'], [name_model], ["red"], fn_2d_cost, [path_png, path_pdf], [".png", ".pdf"])
