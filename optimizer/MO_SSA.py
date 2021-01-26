#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:59, 26/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from copy import deepcopy
from optimizer.root2 import Root2
from numpy.random import choice, uniform, normal, randint
from numpy import array, where, exp, ones, sqrt, sum
from utils.schedule_util import matrix_to_schedule
from uuid import uuid4


class BaseMO_SSA(Root2):
    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub)
        if paras is None:
            paras = {"ST": 0.8, "PD": 0.2, "SD": 0.1}
        self.ST = paras["ST"]  # ST in [0.5, 1.0]
        self.PD = paras["PD"]  # number of producers                        --> In MO: this is the first-front
        self.SD = paras["SD"]  # number of sparrows who perceive the danger --> In MO: this is the last-front

    def get_current_best_worst(self, pop=None):
        fronts, max_rank = self.fast_non_dominated_sort(pop)
        if len(fronts) == 1:
            return choice(fronts[0], 2, replace=False)
        else:
            return fronts[0][randint(0, len(fronts[0]))], fronts[-1][randint(0, len(fronts[-1]))]

    def double_population(self, pop:dict):
        ## Sort population based on fronts
        fronts, max_rank = self.fast_non_dominated_sort(pop)
        pop_new = {}
        for front in fronts:
            for stt in front:
                idx = list(pop.keys())[stt]
                _idx = uuid4().hex
                pop_new[_idx] = deepcopy(pop[idx])
                pop_new[_idx][self.ID_IDX] = _idx
        n1, n2 = len(fronts[0]), len(fronts[-1])
        if n1 == 0 or n1 == self.pop_size:
            n1 = int(self.PD * self.pop_size)
        if n2 == 0 or n2 == self.pop_size:
            n2 = int(self.SD * self.pop_size)

        r2 = uniform()  # R2 in [0, 1], the alarm value, random value
        # Using equation (3) update the sparrow’s location;
        for i in range(0, n1):
            while True:
                if r2 < self.ST:
                    x_new = pop_new[list(pop_new.keys())[i]][self.ID_POS] * exp((i + 1) / (uniform(self.EPSILON, 1) * self.epoch))
                else:
                    x_new = pop_new[list(pop_new.keys())[i]][self.ID_POS] + normal() * ones(self.problem["shape"])
                x_new = self.amend_position_random(x_new)
                schedule = matrix_to_schedule(self.problem, x_new)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    idx = uuid4().hex
                    break
            child = [idx, x_new, fit]
            pop_new[idx] = child

        idx_best, idx_worst = self.get_current_best_worst(pop_new)
        current_best, current_worst = pop_new[list(pop_new.keys())[idx_best]], pop_new[list(pop_new.keys())[idx_worst]]

        # Using equation (4) update the sparrow’s location;
        for i in range(n1, self.pop_size):
            while True:
                if i > int(self.pop_size / 2):
                    x_new = normal() * exp((current_worst[self.ID_POS] - pop_new[list(pop_new.keys())[i]][self.ID_POS]) / (i + 1) ** 2)
                else:
                    x_new = current_best[self.ID_POS] + abs(pop_new[list(pop_new.keys())[i]][self.ID_POS] - current_best[self.ID_POS]) * normal()
                x_new = self.amend_position_random(x_new)
                schedule = matrix_to_schedule(self.problem, x_new)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    idx = uuid4().hex
                    break
            child = [idx, x_new, fit]
            pop_new[idx] = child

        #  Using equation (5) update the sparrow’s location;
        n2_list = choice(list(range(0, self.pop_size)), n2, replace=False)
        for i in n2_list:
            while True:
                child = pop_new[list(pop_new.keys())[i]]
                if i in fronts[0]:
                    x_new = current_best[self.ID_POS] + normal() * abs(child[self.ID_POS] - current_best[self.ID_POS])
                else:
                    dist = sum(sqrt((child[self.ID_FIT] - current_worst[self.ID_FIT])**2))
                    x_new = child[self.ID_POS] + uniform(-1, 1) * (abs(child[self.ID_POS] - current_worst[self.ID_POS]) / (dist + self.EPSILON))
                x_new = self.amend_position_random(x_new)
                schedule = matrix_to_schedule(self.problem, x_new)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    idx = uuid4().hex
                    break
            child = [idx, x_new, fit]
            pop_new[idx] = child

        return {**pop, **pop_new}

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        # Generating new population with double size using SSA-equations
        pop_temp = self.double_population(pop)
        non_dominated_list, max_rank = self.fast_non_dominated_sort(pop_temp)
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
