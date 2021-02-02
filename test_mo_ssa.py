#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 12:10, 01/02/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from sklearn.model_selection import ParameterGrid
from time import time
from copy import deepcopy
from config import OptExp, OptParas
import optimizer
from utils.io_util import load_tasks, load_nodes
from utils.experiment_util import *


def inside_loop(my_model, n_trials, n_timebound, epoch, fe, end_paras):
    for n_tasks in OptExp.N_TASKS:
        tasks = load_tasks(f'{Config.INPUT_DATA}/tasks_{n_tasks}.json')
        problem = deepcopy(my_model['problem'])
        problem["tasks"] = tasks
        problem["n_tasks"] = n_tasks
        problem["shape"] = [len(problem["clouds"]) + len(problem["fogs"]), n_tasks]

        for pop_size in OptExp.POP_SIZE:
            for lb, ub in zip(OptExp.LB, OptExp.UB):
                parameters_grid = list(ParameterGrid(my_model["param_grid"]))
                for paras in parameters_grid:
                    opt = getattr(optimizer, my_model["class"])(problem=problem, pop_size=pop_size, epoch=epoch,
                                                                func_eval=fe, lb=lb, ub=ub, verbose=OptExp.VERBOSE, paras=paras)
                    solutions, g_best, g_best_dict, training_info = opt.train()
                    if Config.TIME_BOUND_KEY:
                        path_results = f'{Config.RESULTS_DATA}/{n_timebound}s/task_{n_tasks}/{Config.METRICS}/{my_model["name"]}/{n_trials}'
                    else:
                        path_results = f'{Config.RESULTS_DATA}/no_time_bound/task_{n_tasks}/{Config.METRICS}/{my_model["name"]}/{n_trials}'
                    Path(path_results).mkdir(parents=True, exist_ok=True)
                    name_paras = f'{epoch}_{pop_size}_{end_paras}'
                    save_experiment_results_multi(solutions, g_best, g_best_dict, training_info, name_paras, path_results)
                    if Config.VISUAL_SAVING:
                        save_visualization_results_multi(solutions, my_model["name"], name_paras, path_results)


def setting_and_running(my_model):
    print(f'Start running: {my_model["name"]}')
    for n_trials in range(OptExp.N_TRIALS):
        if Config.TIME_BOUND_KEY:
            for n_timebound in OptExp.TIME_BOUND_VALUES:
                if Config.MODE == "epoch":
                    for epoch in OptExp.EPOCH:
                        end_paras = f"{epoch}"
                        inside_loop(my_model, n_trials, n_timebound, epoch, None, end_paras)
                elif Config.MODE == "fe":
                    for fe in OptExp.FE:
                        end_paras = f"{fe}"
                        inside_loop(my_model, n_trials, n_timebound, None, fe, end_paras)
        else:
            if Config.MODE == "epoch":
                for epoch in OptExp.EPOCH:
                    end_paras = f"{epoch}"
                    inside_loop(my_model, n_trials, None, epoch, None, end_paras)
            elif Config.MODE == "fe":
                for fe in OptExp.FE:
                    end_paras = f"{fe}"
                    inside_loop(my_model, n_trials, None, None, fe, end_paras)


if __name__ == '__main__':
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
        # {"name": "NSGA-II", "class": "BaseNSGA_II", "param_grid": OptParas.NSGA_II, "problem": problem},
        # {"name": "NSGA-III", "class": "BaseNSGA_III", "param_grid": OptParas.NSGA_III, "problem": problem},
        # {"name": "MO-ALO", "class": "BaseMO_ALO", "param_grid": OptParas.MO_ALO, "problem": problem},
        {"name": "MO-SSA", "class": "BaseMO_SSA", "param_grid": OptParas.MO_SSA, "problem": problem},
        # {"name": "NS-SSA", "class": "BaseNS_SSA", "param_grid": OptParas.MO_SSA, "problem": problem},
    ]

    for my_md in models:
        setting_and_running(my_md)

    print('That took: {} seconds'.format(time() - starttime))

