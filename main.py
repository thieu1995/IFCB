#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 10:15, 11/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from pathlib import Path
import pickle as pkl
from multiprocessing import Pool
from queue import Queue
from copy import deepcopy
from sklearn.model_selection import ParameterGrid
from numpy import array
from pandas import DataFrame

from config import Config
from model.fitness import Fitness
from optimizer.GA import GAEngine
from utils.io_util import load_tasks, load_nodes
from utils.schedule_util import matrix_to_schedule


def save_training_fitness_information(list_fitness, number_tasks, name_mha, name_paras, results_folder_path):
    results_path = f'{results_folder_path}/optimize_process/{name_mha}/{name_paras}'
    Path(results_path).mkdir(parents=True, exist_ok=True)
    fitness_file_path = f'{results_path}/training_{number_tasks}_tasks.csv'
    fitness_df = DataFrame(list_fitness)
    fitness_df.index.name = "epoch"
    fitness_df.to_csv(fitness_file_path, index=True, header=["fitness"])
    if Config.METRICS_NEED_MIN:
        with open(f'{Config.RESULTS_DATA}/summary.txt', 'a+') as f:
            f.write(f'{Config.METRICS}, {number_tasks}, {name_mha}, {name_paras}, {list_fitness[-1]}\n')


def save_experiment_result(problem, solution, name_mha, name_paras, results_folder_path):
    experiment_results_path = f'{results_folder_path}/experiment_results/{name_mha}/{name_paras}'
    Path(experiment_results_path).mkdir(parents=True, exist_ok=True)
    fit_obj = Fitness(problem)
    schedule = matrix_to_schedule(problem, solution[0], solution[1])
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


def __optimize_schedule_with_ga(item):
    pop_size = item['pop_size']
    epoch = item['epoch']
    func_eval = item["func_eval"]
    time_bound = item["time_bound"]
    domain_range = item["domain_range"]
    number_tasks = item["number_tasks"]
    Path(f'{Config.RESULTS_DATA}_{time_bound}').mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(f'{Config.INPUT_DATA}/tasks_{number_tasks}.json')
    problem = deepcopy(item['problem'])
    problem["tasks"] = tasks
    optimizer = GAEngine(problem, pop_size, epoch, func_eval, time_bound, domain_range)
    solution, best_fit, best_fit_list = optimizer.train()
    name_mha = 'ga'
    name_paras = f'{epoch}_{pop_size}'
    results_folder_path = f'{Config.RESULTS_DATA}_{time_bound}/{Config.METRICS}/'
    Path(results_folder_path).mkdir(parents=True, exist_ok=True)

    save_training_fitness_information(best_fit_list, len(tasks), name_mha, name_paras, results_folder_path)
    save_experiment_result(problem, solution, name_mha, name_paras, results_folder_path)


def __optimize_schedule(item):
    # hgso: 'n_clusters': n_clusters
    if item['optimizer'] == 'ga':
        __optimize_schedule_with_ga(item)
    # elif item['optimizer'] == 'pso':
    #     __optimize_schedule_with_pso(item)
    # elif item['optimizer'] == 'woa':
    #     __optimize_schedule_with_woa(item)
    # elif item['optimizer'] == 'hgso':
    #     __optimize_schedule_with_hgso(item)
    # elif item['optimizer'] == 'bla':
    #     __optimize_schedule_with_bla(item)
    else:
        print('|=> We do not support your algorithms')


def optimize_schedule(problem):
    # experiment_case
    num_pool = 1  # 16
    param_grid = {
        'number_tasks': [100],  # list(range(150, 201, 50))
        'pop_size': [50],  # [100]
        'epoch': [10],  # [200]
        'func_eval': [100000],
        'time_bound': [30],  # list(range(0, 10, 1))
        'domain_range': [[-1, 1]],
        'optimizer': ['ga'],
        'problem': [problem]
    }
    queue = Queue()
    for item in list(ParameterGrid(param_grid)):
        queue.put_nowait(item)
    pool = Pool(num_pool)
    pool.map(__optimize_schedule, list(queue.queue))
    pool.close()
    pool.join()
    pool.terminate()


if __name__ == "__main__":
    clouds, fogs, peers = load_nodes(f'{Config.INPUT_DATA}/nodes_2_10_5.json')
    problem = {
        "clouds": clouds,
        "fogs": fogs,
        "peers": peers
    }
    optimize_schedule(problem)



