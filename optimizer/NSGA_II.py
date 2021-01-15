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
from numpy import array, where
from utils.schedule_util import matrix_to_schedule
from uuid import uuid4


class BaseNSGA_II(Root2):

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub)
        if paras is None:
            paras = {"p_c": 0.9, "p_m": 0.05}
        self.p_c = paras["p_c"]
        self.p_m = paras["p_m"]

    def crossover(self, dad, mom):
        r = random()
        if r < self.p_c:
            child = (dad[self.ID_POS] + mom[self.ID_POS]) / 2
            schedule = matrix_to_schedule(self.problem, child)
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                idx = uuid4().hex
            else:
                while True:
                    coef = uniform(0, 1)
                    child = coef * dad[self.ID_POS] + (1 - coef) * mom[self.ID_POS]
                    schedule = matrix_to_schedule(self.problem, child)
                    if schedule.is_valid():
                        fitness = self.Fit.fitness(schedule)
                        idx = uuid4().hex
                        break
            return [idx, child, fitness]
        return dad

    def mutate(self, child):
        while True:
            child_pos = deepcopy(child[self.ID_POS])
            rd_matrix = uniform(self.lb, self.ub, self.problem["shape"])
            child_pos[rd_matrix < self.p_m] = 0
            rd_matrix_new = uniform(self.lb, self.ub, self.problem["shape"])
            rd_matrix_new[rd_matrix >= self.p_m] = 0
            child_pos = child_pos + rd_matrix_new
            schedule = matrix_to_schedule(self.problem, child_pos)
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                idx = uuid4().hex
                break
        return [idx, child_pos, fitness]

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        non_dominated_list = self.fast_non_dominated_sort(pop)
        cdist_lists = []
        for i in range(0, len(non_dominated_list)):
            cdist_lists.append(self.crowding_distance(pop, non_dominated_list[i]))
        pop_temp = deepcopy(pop)

        # Generating offsprings
        while (len(pop_temp) != 2 * self.pop_size):
            stt_dad, stt_mom = choice(list(range(0, self.pop_size)), 2, replace=False)
            idx_dad, idx_mom = list(pop.keys())[stt_dad], list(pop.keys())[stt_mom]
            temp = self.crossover(pop[idx_dad], pop[idx_mom])
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
