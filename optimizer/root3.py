#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 10:36, 28/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from time import time
from config import Config
from sys import exit
from optimizer.root import Root
from numpy import array, inf, zeros, all, any, cumsum, delete, ones, min, max, sum
from numpy.random import uniform
from utils.schedule_util import matrix_to_schedule
from uuid import uuid4
from copy import deepcopy


class Root3(Root):
    ID_IDX = 0
    ID_POS = 1
    ID_FIT = 2

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        self.n_objs = None
        self.WEIGHT = 1e-6

    def create_solution(self):
        while True:
            matrix = uniform(self.lb, self.ub, self.problem["shape"])
            schedule = matrix_to_schedule(self.problem, matrix)
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                idx = uuid4().hex
                break
        return [idx, matrix, fitness]

    def create_opposition_based_pop(self, pop: list):
        size = len(pop)
        pop_opposition = [0 for _ in range(size)]
        for i in range(size):
            while True:
                matrix = self.lb + self.ub - uniform() * pop[i][self.ID_POS]
                schedule = matrix_to_schedule(self.problem, matrix)
                if schedule.is_valid():
                    fitness = self.Fit.fitness(schedule)
                    idx = uuid4().hex
                    break
            pop_opposition[i] = [idx, matrix, fitness]
        return pop_opposition

    def dominates(self, fit_a, fit_b):
        return all(fit_a <= fit_b) and any(fit_a < fit_b)

    def find_dominates_list(self, pop:list):
        size = len(pop)
        list_dominated = zeros(size)  # 0: non-dominated, 1: dominated by someone
        list_fit = array([solution[self.ID_FIT] for solution in pop])
        for i in range(0, size):
            list_dominated[i] = 0
            for j in range(0, i):
                if any(list_fit[i] != list_fit[j]):
                    if self.dominates(list_fit[i], list_fit[j]):
                        list_dominated[j] = 1
                    elif self.dominates(list_fit[j], list_fit[i]):
                        list_dominated[i] = 1
                        break
                else:
                    list_dominated[j] = 1
                    list_dominated[i] = 1
        return list_dominated

    def update_pop_archive(self, pop_archive):
        size = len(pop_archive)
        list_dominated = self.find_dominates_list(pop_archive)
        pop = []
        fits = []
        ## If all non-dominated --> Save all and remove some duplicate one
        if sum(list_dominated) == 0 or sum(list_dominated) == len(list_dominated):
            for i, solution in enumerate(pop_archive):
                flag = False
                for j in range(i + 1, size):
                    if all(pop_archive[i][self.ID_FIT] == pop_archive[j][self.ID_FIT]):
                        flag = True
                        break
                if not flag:
                    pop.append(solution)
                    fits.append(solution[self.ID_FIT])
            return pop, fits
        ## Else just save non-dominated solutions
        for i in range(0, size):
            if list_dominated[i] == 0:
                pop.append(pop_archive[i])
                fits.append(pop_archive[i][self.ID_FIT])
        return pop, fits

    def neighbourhood_ranking(self, archive_fits, division=20):
        size = len(archive_fits)
        my_min = min(archive_fits, axis=0)
        my_max = max(archive_fits, axis=0)
        r = (my_max - my_min) / division
        ranks = zeros(size)

        for i in range(0, size):
            ranks[i] = 0
            for j in range(0, size):
                # Calculate a number of neighbourhood in all dimensions using radius r
                ## More neighbourhood -> Higher ranking --> Should be removed (because we are looking for rare points - meaning less neighbourhood
                if all(abs(archive_fits[j] - archive_fits[i]) < r):
                    ranks[i] += 1
        return ranks

    def roulette_wheel_selection(self, weights):
        accumulation = cumsum(weights)
        p = uniform() * accumulation[-1]
        chosen_index = -1
        for idx in range(0, len(accumulation)):
            if accumulation[idx] > p:
                chosen_index = idx
                break
        return chosen_index

    def handle_full_archive(self, pop_archive, archive_fits, archive_ranks, archive_max_size):
        drop_size = len(pop_archive) - archive_max_size
        for idx in range(0, drop_size):
            index = self.roulette_wheel_selection(archive_ranks)
            del pop_archive[index]              # list
            del archive_fits[index]             # list
            archive_ranks = delete(archive_ranks, index)    # nd-array
        return pop_archive, archive_fits, archive_ranks

    def find_global_best(self, pop:list):
        global_best = [0, zeros(self.problem["shape"]), inf * ones(self.n_objs)]
        for solution in pop:
            if self.dominates(solution[self.ID_FIT], global_best[self.ID_FIT]):
                global_best = deepcopy(solution)
        return global_best

    def update_global_best(self, global_best, current_best):
        if self.dominates(global_best[self.ID_FIT], current_best[self.ID_FIT]):
            return current_best
        elif self.dominates(current_best[self.ID_FIT], global_best[self.ID_FIT]):
            return global_best
        else:
            return global_best if uniform() <= 0.5 else current_best

    def evolve2(self, pop:list, pop_archive:list, fe_mode=None, epoch=None, g_best=None):
        pass

    def train(self):
        time_counting = time()
        if Config.METRICS == "pareto" and self.__class__.__name__ not in Config.MULTI_OBJECTIVE_SUPPORTERS:
            print(f'Method: {self.__class__.__name__} doesn"t support pareto-front fitness function type')
            exit()
        if self.verbose:
            print(f'Start training by: {self.__class__.__name__} algorithm, fitness type: {Config.METRICS}')

        pop = [self.create_solution() for _ in range(self.pop_size)]
        self.n_objs = len(pop[0][self.ID_FIT])
        # The archive population (at first: its size is the same size with the population, then its changing iteratively)
        pop_archive = self.create_opposition_based_pop(pop)
        g_best_dict = {}  # Fronts[0] by iterations
        # The best solution which is an archive member and it in the least populated area as an attractant to improve coverage
        global_best = self.find_global_best(pop)            # Helping AntLion only, the results we want is the first-pareto-front

        training_info = {"Epoch": [], "FrontSize": [], "ArchiveSize": [], "Time": []}
        time_bound_start = time()
        time_bound_log = f'with time-bound: {Config.TIME_BOUND_VALUE} seconds.' if Config.TIME_BOUND_KEY else "without time-bound."
        if Config.MODE == 'epoch':
            if self.verbose:
                print(f'Training by: epoch (mode) with: {self.epoch} epochs, {time_bound_log}')
            for epoch in range(self.epoch):
                time_epoch_start = time()

                pop = self.evolve2(pop, pop_archive, None, epoch, global_best)

                pop_new = pop_archive + pop
                pop_archive, archive_fits = self.update_pop_archive(pop_new)
                if len(pop_archive) > self.pop_size:
                    archive_ranks = self.neighbourhood_ranking(archive_fits)
                    pop_archive, archive_fits, archive_ranks = self.handle_full_archive(pop_archive, archive_fits, archive_ranks, self.pop_size)
                else:
                    archive_ranks = self.neighbourhood_ranking(archive_fits)

                ## Choose the archive member in the least populated area as an attractant to improve coverage
                idx = self.roulette_wheel_selection(1.0 / archive_ranks)
                idx = 0 if idx == -1 else idx
                current_best = pop_archive[idx]
                global_best = self.update_global_best(global_best, current_best)

                pop_new = pop_archive + pop
                non_dominated_list = self.find_dominates_list(pop_new)
                current_best = []
                for idx, value in enumerate(non_dominated_list):
                    if value == 0:
                        current_best.append(pop_new[idx][self.ID_FIT])
                g_best_dict[epoch + 1] = array(current_best)

                time_epoch_end = time() - time_epoch_start
                training_info = self.adding_element_to_dict(training_info, ["Epoch", "FrontSize", "ArchiveSize", "Time"],
                                                            [epoch + 1, len(non_dominated_list), len(pop_archive), time_epoch_end])
                if self.verbose:
                    print(f'Epoch: {epoch+1}, Front0 size: {len(non_dominated_list)}, Archive size: {len(pop_archive)}, '
                          f'First front0 fit: {g_best_dict[epoch][0]}, with time: {time_epoch_end:.2f} seconds')
                if Config.TIME_BOUND_KEY:
                    if time() - time_bound_start >= Config.TIME_BOUND_VALUE:
                        print('====== Over time for training ======')
                        break
            solutions = {}
            g_best = []
            for idx, value in enumerate(non_dominated_list):
                if value == 0:
                    g_best.append(pop_new[idx][self.ID_FIT])
            time_counting = time() - time_counting
            training_info = self.adding_element_to_dict(training_info, ["Epoch", "FrontSize", "ArchiveSize", "Time"],
                                                        [epoch + 1, len(non_dominated_list), len(pop_archive), time_counting])
            return solutions, array(g_best), g_best_dict, training_info

