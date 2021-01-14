#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:41, 11/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from sklearn.model_selection import ParameterGrid
import multiprocessing
from time import time
from pathlib import Path
import pickle as pkl
from copy import deepcopy
from numpy import array
from pandas import DataFrame

from config import Config, OptParas, OptExp
from model.fitness import Fitness
from utils.io_util import load_tasks, load_nodes
from utils.schedule_util import matrix_to_schedule
import optimizer


def save_training_fitness_information(list_fitness, number_tasks, name_mha, name_paras, results_folder_path):
    results_path = f'{results_folder_path}/optimize_process/{name_mha}/{name_paras}'
    Path(results_path).mkdir(parents=True, exist_ok=True)
    fitness_file_path = f'{results_path}/training_{number_tasks}_tasks.csv'
    fitness_df = DataFrame(list_fitness)
    fitness_df.index.name = "epoch"
    fitness_df.to_csv(fitness_file_path, index=True, header=["fitness"])
    if Config.METRICS_NEED_MIN_OBJECTIVE_VALUES:
        with open(f'{Config.RESULTS_DATA}/summary.txt', 'a+') as f:
            f.write(f'{Config.METRICS}, {number_tasks}, {name_mha}, {name_paras}, {list_fitness[-1]}\n')


def save_experiment_result(problem, solution, name_mha, name_paras, results_folder_path):
    experiment_results_path = f'{results_folder_path}/experiment_results/{name_mha}/{name_paras}'
    Path(experiment_results_path).mkdir(parents=True, exist_ok=True)
    fit_obj = Fitness(problem)
    schedule = matrix_to_schedule(problem, solution)
    power = fit_obj.calc_power_consumption(schedule)
    latency = fit_obj.calc_latency(schedule)
    cost = fit_obj.calc_cost(schedule)
    experiment_results = array([[power, latency, cost]])
    experiment_results_df = DataFrame(experiment_results)
    file_name = f'{experiment_results_path}/{len(problem["tasks"])}_tasks'
    experiment_results_df.index.name = "Trial"
    experiment_results_df.to_csv(f'{file_name}.csv', header=["Power", "Latency", "Cost"], index=True)
    schedule_object_save_path = open(f'{file_name}.pickle', 'wb')
    pkl.dump(schedule, schedule_object_save_path)
    schedule_object_save_path.close()


def inside_loop(my_model, n_trials, n_timebound):
    for n_tasks in OptExp.N_TASKS:
        Path(f'{Config.RESULTS_DATA}_{n_trials}').mkdir(parents=True, exist_ok=True)
        tasks = load_tasks(f'{Config.INPUT_DATA}/tasks_{n_tasks}.json')
        problem = deepcopy(my_model['problem'])
        problem["tasks"] = tasks
        for pop_size in OptExp.POP_SIZE:
            for dr in OptExp.DOMAIN_RANGE:
                parameters_grid = list(ParameterGrid(my_model["param_grid"]))
                if Config.MODE == "epoch":
                    for epoch in OptExp.EPOCH:
                        for paras in parameters_grid:
                            name_paras = f'{pop_size}_{epoch}'
                            opt = getattr(optimizer, my_model["name"])(problem=problem, pop_size=pop_size, epoch=epoch,
                                                                       func_eval=None, time_bound=n_timebound, domain_range=dr, paras=paras)
                            solution, best_fit, best_fit_list = opt.train()
                elif Config.MODE == "fe":
                    for fe in OptExp.FE:
                        for paras in parameters_grid:
                            name_paras = f'{pop_size}_{fe}'
                            opt = getattr(optimizer, my_model["name"])(problem=problem, pop_size=pop_size, epoch=None,
                                                                       func_eval=fe, time_bound=n_timebound, domain_range=dr, paras=paras)
                            solution, best_fit, best_fit_list = opt.train()

                if Config.TIME_BOUND:
                    results_folder_path = f'{Config.RESULTS_DATA}_{n_timebound}s/{Config.METRICS}/'
                else:
                    results_folder_path = f'{Config.RESULTS_DATA}_no_time_bound/{Config.METRICS}/'
                Path(results_folder_path).mkdir(parents=True, exist_ok=True)

                save_training_fitness_information(best_fit_list, len(tasks), my_model["name"], name_paras, results_folder_path)
                save_experiment_result(problem, solution, my_model["name"], name_paras, results_folder_path)



def setting_and_running(my_model):
    print(f'Start running: {my_model["name"]}')
    for n_trials in OptExp.N_TRIALS:
        if Config.TIME_BOUND:
            for n_timebound in OptExp.TIME_BOUND_VALUES:
                inside_loop(my_model, n_trials, n_timebound)
        else:
            inside_loop(my_model, n_trials, None)


if __name__ == '__main__':
    starttime = time()
    clouds, fogs, peers = load_nodes(f'{Config.INPUT_DATA}/nodes_2_10_5.json')
    problem = {
        "clouds": clouds,
        "fogs": fogs,
        "peers": peers
    }
    models = [
        # {"name": "BaseGA", "param_grid": OptParas.GA, "problem": problem},
        # {"name": "BasePSO", "param_grid": OptParas.PSO, "problem": problem},
        # {"name": "BaseWOA", "param_grid": OptParas.WOA, "problem": problem},
        {"name": "BaseEO", "param_grid": OptParas.EO, "problem": problem},
    ]

    processes = []
    for my_md in models:
        p = multiprocessing.Process(target=setting_and_running, args=(my_md,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print('That took: {} seconds'.format(time() - starttime))

