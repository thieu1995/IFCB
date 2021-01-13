#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:32, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from optimizer.root import Root
from numpy.random import uniform, random, choice
from numpy import exp, cos, pi
from utils.schedule_util import matrix_to_schedule


class BaseWOA(Root):

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, time_bound=None, domain_range=None, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, time_bound, domain_range)
        if paras is None:
            paras = {"p": 0.5, "b": 1.0}
        self.p = paras["p"]
        self.b = paras["b"]

    def evolve(self, pop, fe_mode=None, epoch=None, g_best=None):
        a = 2 - 2 * epoch / (self.epoch - 1)    # linearly decreased from 2 to 0
        for i in range(self.pop_size):
            r = random()
            A = 2 * a * r - a
            C = 2 * r
            l = uniform(-1, 1)

            if uniform() < self.p:
                if abs(A) < 1:
                    while True:
                        child = []
                        for j in range(len(pop[i][self.ID_POS])):
                            D = g_best[self.ID_POS][j] - A * abs(C * g_best[self.ID_POS][j] - pop[i][self.ID_POS][j])
                            child.append(D)
                        schedule = matrix_to_schedule(self.problem, child[0], child[1])
                        if schedule.is_valid():
                            fitness = self.Fit.fitness(schedule)
                            break
                else:
                    while True:
                        child = []
                        id_rand = choice(list(set(range(0, self.pop_size)) - {i}))   # select random 1 position in pop
                        for j in range(len(pop[i][self.ID_POS])):
                            D = pop[id_rand][self.ID_POS][j] - A * abs(C * pop[id_rand][self.ID_POS][j] - pop[i][self.ID_POS][j])
                            child.append(D)
                        schedule = matrix_to_schedule(self.problem, child[0], child[1])
                        if schedule.is_valid():
                            fitness = self.Fit.fitness(schedule)
                            break
            else:
                while True:
                    child = []
                    for j in range(len(pop[i][self.ID_POS])):
                        D1 = abs(g_best[self.ID_POS][j] - pop[i][self.ID_POS][j])
                        D = D1 * exp(self.b * l) * cos(2 * pi * l) + g_best[self.ID_POS][j]
                        child.append(D)
                    schedule = matrix_to_schedule(self.problem, child[0], child[1])
                    if schedule.is_valid():
                        fitness = self.Fit.fitness(schedule)
                        break
            pop[i] = [child, fitness]

        if fe_mode is None:
            return pop
        else:
            counter = 2 * self.pop_size  # pop_new + pop_mutation operations
            return pop, counter
