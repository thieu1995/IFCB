#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 12:27, 15/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from copy import deepcopy
from config import Config
from optimizer.root import Root
from numpy.random import uniform, randint, normal
from utils.schedule_util import matrix_to_schedule


class BaseAEO(Root):

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub)

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        # Sorted population in the descending order of the function fitness value
        if Config.METRICS in Config.METRICS_MAX:
            pop = sorted(pop, key=lambda item: item[self.ID_FIT])
        else:
            pop = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
        pop_new = deepcopy(pop)
        ## Production - Update the worst solution
        # Eq. 2, 3, 1
        a = (1.0 - epoch / self.epoch) * uniform()
        while True:
            child = (1 - a) * pop[-1][self.ID_POS] + a * uniform(self.lb, self.ub, pop[-1][self.ID_POS].shape)
            schedule = matrix_to_schedule(self.problem, child)
            if schedule.is_valid():
                fit = self.Fit.fitness(schedule)
                break
        pop_new[0] = [child, fit]

        ## Consumption
        for i in range(2, self.pop_size):
            while True:
                rand = uniform()
                # Eq. 4, 5, 6
                v1 = normal(0, 1)
                v2 = normal(0, 1)
                c = 0.5 * v1 / abs(v2)  # Consumption factor
                j = randint(1, i)
                ### Herbivore
                if rand < 1.0 / 3:
                    child = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[0][self.ID_POS])  # Eq. 6
                ### Carnivore
                elif 1.0 / 3 <= rand and rand <= 2.0 / 3:
                    child = pop[i][self.ID_POS] + c * (pop[i][self.ID_POS] - pop[j][self.ID_POS])  # Eq. 7
                ### Omnivore
                else:
                    r2 = uniform()
                    child = pop[i][self.ID_POS] + c * (r2 * (pop[i][self.ID_POS] - pop[0][self.ID_POS]) + (1 - r2) * (pop[i][self.ID_POS] - pop[j][self.ID_POS]))
                child = self.amend_position_random(child)
                schedule = matrix_to_schedule(self.problem, child)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    break
            pop_new[i] = [child, fit]
        ## Update old population
        pop = self.update_old_population(pop, pop_new)

        ## find current best used in decomposition
        current_best = self.get_current_best(pop)

        ## Decomposition
        ### Eq. 10, 11, 12, 9
        for i in range(0, self.pop_size):
            while True:
                r3 = uniform()
                d = 3 * normal(0, 1)
                e = r3 * randint(1, 3) - 1
                h = 2 * r3 - 1
                child = current_best[self.ID_POS] + d * (e * current_best[self.ID_POS] - h * pop[i][self.ID_POS])
                child = self.amend_position_random(child)
                schedule = matrix_to_schedule(self.problem, child)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    break
            pop_new[i] = [child, fit]
        ## Update old population
        pop = self.update_old_population(pop, pop_new)

        if fe_mode is None:
            return pop
        else:
            counter = 2 * self.pop_size  # pop_new + pop_mutation operations
            return pop, counter
