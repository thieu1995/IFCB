#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:35, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import array, ptp, where, logical_and
from numpy.random import uniform
from time import time
from copy import deepcopy
from config import Config
from model.fitness import Fitness
from utils.schedule_util import matrix_to_schedule
from sys import exit


class Root:
    ID_POS = 0
    ID_FIT = 1

    EPSILON = 10E-10

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None):
        self.problem = problem
        self.pop_size = pop_size
        self.epoch = epoch
        self.func_eval = func_eval
        self.lb = lb
        self.ub = ub
        self.Fit = Fitness(problem)

    def create_solution(self):
        while True:
            matrix = uniform(self.lb, self.ub, self.problem["shape"])
            schedule = matrix_to_schedule(self.problem, matrix)
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                break
        return [matrix, fitness]        # [solution, fit]

    def early_stopping(self, array, patience=5):
        if patience <= len(array) - 1:
            value = array[len(array) - patience]
            arr = array[len(array) - patience + 1:]
            check = 0
            for val in arr:
                if val < value:
                    check += 1
            if check != 0:
                return False
            return True
        raise ValueError

    def get_index_roulette_wheel_selection(self, list_fitness: list):
        """ It can handle negative also. Make sure your list fitness is 1D-numpy array"""
        list_fitness = array(list_fitness)
        scaled_fitness = (list_fitness - min(list_fitness)) / ptp(list_fitness)
        minimized_fitness = 1.0 - scaled_fitness
        total_sum = sum(minimized_fitness)
        r = uniform(low=0, high=total_sum)
        for idx, f in enumerate(minimized_fitness):
            r = r + f
            if r > total_sum:
                return idx

    def amend_position_random(self, position=None):
        return where(logical_and(self.lb <= position, position <= self.ub), position, uniform(self.lb, self.ub))

    def get_current_worst(self, pop=None):
        if isinstance(pop, dict):
            pop_temp = deepcopy(pop.values())
        elif isinstance(pop, list):
            pop_temp = deepcopy(pop)
        else:
            exit()
        if Config.METRICS in Config.METRICS_MAX:
            current_worst = min(pop_temp, key=lambda x: x[self.ID_FIT])
        else:
            current_worst = max(pop_temp, key=lambda x: x[self.ID_FIT])
        return deepcopy(current_worst)

    def get_current_best(self, pop=None):
        if isinstance(pop, dict):
            pop_temp = deepcopy(pop.values())
        elif isinstance(pop, list):
            pop_temp = deepcopy(pop)
        else:
            exit()
        if Config.METRICS in Config.METRICS_MAX:
            current_best = max(pop_temp, key=lambda x: x[self.ID_FIT])
        else:
            current_best = min(pop_temp, key=lambda x: x[self.ID_FIT])
        return deepcopy(current_best)

    def update_old_solution(self, old_solution, new_solution):
        if Config.METRICS in Config.METRICS_MAX:
            if new_solution[self.ID_FIT] > old_solution[self.ID_FIT]:
                return new_solution
        else:
            if new_solution[self.ID_FIT] < old_solution[self.ID_FIT]:
                return new_solution
        return old_solution

    def update_old_population(self, pop_old:list, pop_new:list):
        for i in range(0, self.pop_size):
            if Config.METRICS in Config.METRICS_MAX:
                if pop_new[i][self.ID_FIT] > pop_old[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
            else:
                if pop_new[i][self.ID_FIT] < pop_old[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
        return pop_old

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        pass

    def train(self):
        if Config.METRICS == "pareto" and self.__class__.__name__ not in Config.MULTI_OBJECTIVE_SUPPORTERS:
            print(f'Method: {self.__class__.__name__} doesn"t support pareto-front fitness function type')
            exit()
        print(f'Start training by: {self.__class__.__name__} algorithm, fitness type: {Config.METRICS}')
        pop = [self.create_solution() for _ in range(self.pop_size)]
        if Config.METRICS in Config.METRICS_MAX:
            g_best = max(pop, key=lambda x: x[self.ID_FIT])
        else:
            g_best = min(pop, key=lambda x: x[self.ID_FIT])
        g_best_list = [g_best[self.ID_FIT]]
        time_bound_start = time()
        time_bound_log = "without time-bound."
        if Config.TIME_BOUND_KEY:
            time_bound_log = f'with time-bound: {Config.TIME_BOUND_VALUE} seconds.'
        if Config.MODE == 'epoch':
            print(f'Training by: epoch (mode) with: {self.epoch} epochs, {time_bound_log}')
            for epoch in range(self.epoch):
                time_epoch_start = time()
                pop = self.evolve(pop, None, epoch, g_best)
                if Config.METRICS in Config.METRICS_MAX:
                    current_best = max(pop, key=lambda x: x[self.ID_FIT])
                    if current_best[self.ID_FIT] > g_best_list[-1]:
                        g_best = deepcopy(current_best)
                else:
                    current_best = min(pop, key=lambda x: x[1])
                    if current_best[self.ID_FIT] < g_best_list[-1]:
                        g_best = deepcopy(current_best)
                g_best_list.append(g_best[self.ID_FIT])
                time_epoch_end = time() - time_epoch_start
                print(f'Current best fit {current_best[self.ID_FIT]:.4f}, '
                      f'Global best fit {g_best[self.ID_FIT]:.4f}, '
                      f'Epoch {epoch + 1} with time: {time_epoch_end:.2f} seconds')
                if Config.TIME_BOUND_KEY:
                    if time() - time_bound_start >= Config.TIME_BOUND_VALUE:
                        print('====== Over time for training ======')
                        break
            return g_best[0], g_best[1], array(g_best_list)
        elif Config.MODE == "fe":
            print(f'Training by: function evalution (mode) with: {self.func_eval} FE, {time_bound_log}')
            fe_counter = 0
            time_fe_start = time()
            while fe_counter < self.func_eval:
                pop, counter = self.evolve(pop, Config.MODE, None, g_best)
                if Config.METRICS in Config.METRICS_MAX:
                    current_best = max(pop, key=lambda x: x[self.ID_FIT])
                    if current_best[self.ID_FIT] > g_best_list[-1]:
                        g_best = deepcopy(current_best)
                else:
                    current_best = min(pop, key=lambda x: x[1])
                    if current_best[self.ID_FIT] < g_best_list[-1]:
                        g_best = deepcopy(current_best)
                g_best_list.append(g_best[self.ID_FIT])
                fe_counter += counter
                time_fe_end = time() - time_fe_start
                print(f'Current best fit {current_best[self.ID_FIT]:.4f}, '
                      f'Global best fit {g_best[self.ID_FIT]:.4f}, '
                      f'FE {fe_counter} with time: {time_fe_end:.2f} seconds')
                if Config.TIME_BOUND_KEY:
                    if time() - time_bound_start >= Config.TIME_BOUND_VALUE:
                        print('====== Over time for training ======')
                        break
            return g_best[0], g_best[1], array(g_best_list)

