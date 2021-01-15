#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 12:26, 15/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from optimizer.root import Root
from numpy.random import uniform, normal
from numpy import exp, ones
from utils.schedule_util import matrix_to_schedule
from config import Config
from copy import deepcopy


class BaseSSA(Root):

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub)
        if paras is None:
            paras = {"ST": 0.8, "PD": 0.2, "SD": 0.1}
        self.ST = paras["ST"]  # ST in [0.5, 1.0]
        self.PD = paras["PD"]  # number of producers
        self.SD = paras["SD"]  # number of sparrows who perceive the danger
        self.n1 = int(self.PD * self.pop_size)
        self.n2 = int(self.SD * self.pop_size)

    def evolve(self, pop, fe_mode=None, epoch=None, g_best=None):
        r2 = uniform()  # R2 in [0, 1], the alarm value, random value
        # Using equation (3) update the sparrow’s location;
        for i in range(0, self.n1):
            while True:
                if r2 < self.ST:
                    child = pop[i][self.ID_POS] * exp((i + 1) / (uniform() * self.epoch))
                else:
                    child = pop[i][self.ID_POS] + normal() * ones(self.problem["shape"])
                child = self.amend_position_random(child)
                schedule = matrix_to_schedule(self.problem, child)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    break
            pop[i] = self.update_old_solution(pop[i], [child, fit])

        child_p = deepcopy(self.get_current_best(pop[:self.n1]))
        worst = deepcopy(self.get_current_worst(pop))

        # Using equation (4) update the sparrow’s location;
        for i in range(self.n1, self.pop_size):
            while True:
                if i > int(self.pop_size / 2):
                    x_new = normal() * exp((worst[self.ID_POS] - pop[i][self.ID_POS]) / (i + 1) ** 2)
                else:
                    x_new = child_p[self.ID_POS] + abs(pop[i][self.ID_POS] - child_p[self.ID_POS]) * normal()
                x_new = self.amend_position_random(x_new)
                schedule = matrix_to_schedule(self.problem, x_new)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    break
            pop[i] = self.update_old_solution(pop[i], [x_new, fit])

        #  Using equation (5) update the sparrow’s location;
        for i in range(0, self.n2):
            while True:
                if Config.METRICS in Config.METRICS_MAX:
                    if pop[i][self.ID_FIT] < g_best[self.ID_FIT]:
                        x_new = g_best[self.ID_POS] + normal() * abs(pop[i][self.ID_POS] - g_best[self.ID_POS])
                    else:
                        x_new = pop[i][self.ID_POS] + uniform(-1, 1) * \
                                (abs(pop[i][self.ID_POS] - worst[self.ID_POS]) / (pop[i][self.ID_FIT] - worst[self.ID_FIT] + self.EPSILON))
                else:
                    if pop[i][self.ID_FIT] > g_best[self.ID_FIT]:
                        x_new = g_best[self.ID_POS] + normal() * abs(pop[i][self.ID_POS] - g_best[self.ID_POS])
                    else:
                        x_new = pop[i][self.ID_POS] + uniform(-1, 1) * \
                                (abs(pop[i][self.ID_POS] - worst[self.ID_POS]) / (pop[i][self.ID_FIT] - worst[self.ID_FIT] + self.EPSILON))
                x_new = self.amend_position_random(x_new)
                schedule = matrix_to_schedule(self.problem, x_new)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    break
            pop[i] = self.update_old_solution(pop[i], [x_new, fit])

        if fe_mode is None:
            return pop
        else:
            counter = 2 * self.n1 + self.n2  # pop_new + pop_mutation operations
            return pop, counter

