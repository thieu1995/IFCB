#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 03:12, 14/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from copy import deepcopy
from config import Config
from optimizer.root2 import Root2
from numpy.random import uniform, random, choice
from numpy import array, zeros, argmin, inf, max, min, where
from utils.schedule_util import matrix_to_schedule
from uuid import uuid4


class BaseNSGA_II(Root2):

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, time_bound=None, domain_range=None, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, time_bound, domain_range)
        if paras is None:
            paras = {"p_c": 0.9, "p_m": 0.05}
        self.p_c = paras["p_c"]
        self.p_m = paras["p_m"]

    def crossover(self, dad, mom):
        r = random()
        child = []
        if r < self.p_c:
            while True:
                for i in range(len(dad[self.ID_POS])):
                    child.append((dad[self.ID_POS][i] + mom[self.ID_POS][i]) / 2)
                schedule = matrix_to_schedule(self.problem, child)
                if schedule.is_valid():
                    fitness = self.Fit.fitness(schedule)
                    idx = uuid4().hex
                    break
            return [idx, child, fitness]  # [solution, fit]
        return dad

    def mutate(self, child):
        child_new = []
        while True:
            for j in range(len(child[self.ID_POS])):
                sol_part_temp = child[self.ID_POS][j]
                for k_row in range(sol_part_temp.shape[0]):
                    for k_col in range(sol_part_temp.shape[1]):
                        if uniform() < self.p_m:
                            sol_part_temp[k_row][k_col] = uniform(self.domain_range[0], self.domain_range[1])
                child_new.append(sol_part_temp)
            schedule = matrix_to_schedule(self.problem, child_new)
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                idx = uuid4().hex
                break
        return [idx, child_new, fitness]

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        non_dominated_list = self.fast_non_dominated_sort(pop)
        cdist_lists = []
        for i in range(0, len(non_dominated_list)):
            cdist_lists.append(self.crowding_distance(pop, non_dominated_list[i]))
        pop_temp = deepcopy(pop)

        # Generating offsprings
        while (len(pop_temp) != 2 * self.pop_size):
            id_dad, id_mom = choice(list(range(0, self.pop_size)), 2, replace=False)
            temp = self.crossover(list(pop.items())[id_dad][1], list(pop.items())[id_mom][1])
            temp = self.mutate(temp)
            pop_temp[temp[self.ID_IDX]] = temp

        non_dominated_list = self.fast_non_dominated_sort(pop_temp)
        cdist_lists = []
        for i in range(0, len(non_dominated_list)):
            cdist_lists.append(self.crowding_distance(pop_temp, non_dominated_list[i]))

        pop = {}
        for i in range(0, len(non_dominated_list)):
            current_NDS_list = array(non_dominated_list[i])
            NDS_list_child = [where(current_NDS_list == current_NDS_list[j]) for j in range(0, len(current_NDS_list))]
            front22 = self.sort_by_values(NDS_list_child, cdist_lists[i])
            front = [current_NDS_list[front22[j]] for j in range(0, len(current_NDS_list))]
            front.reverse()
            for idx in front:
                idx_real = list(pop_temp.keys())[idx]
                pop[idx_real] = pop_temp[idx_real]
                if len(pop) == self.pop_size:
                    break
            if len(pop) == self.pop_size:
                break
        return pop
