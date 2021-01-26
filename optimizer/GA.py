#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:32, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from copy import deepcopy
from config import Config
from optimizer.root import Root
from numpy.random import uniform
from utils.schedule_util import matrix_to_schedule


class BaseGA(Root):

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub)
        if paras is None:
            paras = {"p_c": 0.9, "p_m": 0.05}
        self.p_c = paras["p_c"]
        self.p_m = paras["p_m"]

    def crossover(self, dad, mom):
        if uniform() < self.p_c:
            child = (dad[self.ID_POS] + mom[self.ID_POS]) / 2
            schedule = matrix_to_schedule(self.problem, child)
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
            else:
                while True:
                    coef = uniform(0, 1)
                    child = coef*dad[self.ID_POS] + (1-coef)*mom[self.ID_POS]
                    schedule = matrix_to_schedule(self.problem, child)
                    if schedule.is_valid():
                        fitness = self.Fit.fitness(schedule)
                        break
            return [child, fitness]
        else:
            if Config.METRICS in Config.METRICS_MAX:
                return mom if dad[self.ID_FIT] < mom[self.ID_FIT] else dad
            else:
                return dad if dad[self.ID_FIT] < mom[self.ID_FIT] else mom

    def select(self, pop):
        pop_new = []
        while len(pop_new) < self.pop_size:
            fit_list = [item[self.ID_FIT] for item in pop]
            dad_index = self.get_index_roulette_wheel_selection(fit_list)
            mom_index = self.get_index_roulette_wheel_selection(fit_list)
            while dad_index == mom_index:
                mom_index = self.get_index_roulette_wheel_selection(fit_list)
            dad = pop[dad_index]
            mom = pop[mom_index]
            sol_new = self.crossover(dad, mom)
            pop_new.append(sol_new)
        return pop_new

    def mutate(self, pop):
        for i in range(self.pop_size):
            while True:
                child = deepcopy(pop[i][self.ID_POS])
                rd_matrix = uniform(self.lb, self.ub, self.problem["shape"])
                child[rd_matrix < self.p_m] = 0
                rd_matrix_new = uniform(self.lb, self.ub, self.problem["shape"])
                rd_matrix_new[rd_matrix >= self.p_m] = 0
                child = child + rd_matrix_new
                schedule = matrix_to_schedule(self.problem, child)
                if schedule.is_valid():
                    fitness = self.Fit.fitness(schedule)
                    break
            pop[i] = [child, fitness]
        return pop

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        pop = self.select(pop)
        pop = self.mutate(pop)
        if fe_mode is None:
            return pop
        else:
            counter = 2*self.pop_size   # pop_new + pop_mutation operations
            return pop, counter


