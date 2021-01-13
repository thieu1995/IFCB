#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 01:49, 14/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from config import Config
from optimizer.root import Root
from numpy.random import uniform, randint
from numpy import exp, mean, sign, ones
from utils.schedule_util import matrix_to_schedule


class BaseEO(Root):

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, time_bound=None, domain_range=None, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, time_bound, domain_range)
        if paras is None:
            paras = {"V": 1.0, "a1": 2.0, "a2": 1.0, "GP": 0.5}
        self.V = paras["V"]
        self.a1 = paras["a1"]
        self.a2 = paras["a2"]
        self.GP = paras["GP"]

    def make_equilibrium_pool(self, list_equilibrium=None, c_eq5=None):  # make equilibrium pool
        c_mean = []
        for idx in range(len(list_equilibrium[0][self.ID_POS])):
            pos_list = [item[self.ID_POS][idx] for item in list_equilibrium]
            pos_mean = mean(pos_list, axis=0)
            c_mean.append(pos_mean)
        schedule = matrix_to_schedule(self.problem, c_mean[0], c_mean[1])
        if schedule.is_valid():
            fit = self.Fit.fitness(schedule)
            list_equilibrium.append([c_mean, fit])
        else:
            list_equilibrium.append(c_eq5)
        return list_equilibrium

    def evolve(self, pop, fe_mode=None, epoch=None, g_best=None):
        if Config.METRICS in Config.METRICS_MAX:
            pop_sorted = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
        else:
            pop_sorted = sorted(pop, key=lambda item: item[self.ID_FIT])
        c_eq_list, c_eq5 = pop_sorted[:4], pop_sorted[4]
        c_pool = self.make_equilibrium_pool(c_eq_list, c_eq5)

        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)  # Eq. 9
        for i in range(0, self.pop_size):
            while True:
                child = pop[i][self.ID_POS]
                c_eq = c_pool[randint(0, len(c_pool))][self.ID_POS]  # random selection 1 of candidate from the pool
                for j in range(len(child)):
                    lamda = uniform(0, 1, child[j].shape)  # lambda in Eq. 11
                    r = uniform(0, 1, child[j].shape)  # r in Eq. 11
                    f = self.a1 * sign(r - 0.5) * (exp(-lamda * t) - 1.0)  # Eq. 11
                    r1 = uniform()
                    r2 = uniform()  # r1, r2 in Eq. 15
                    gcp = 0.5 * r1 * ones(child[j].shape) * (r2 >= self.GP)  # Eq. 15
                    g0 = gcp * (c_eq[j] - lamda * pop[i][self.ID_POS][j])  # Eq. 14
                    g = g0 * f  # Eq. 13
                    temp = c_eq[j] + (pop[i][self.ID_POS][j] - c_eq[j]) * f + (g * self.V / lamda) * (1.0 - f)  # Eq. 16
                    temp = self.amend_position_random(temp)
                    child.append(temp)
                schedule = matrix_to_schedule(self.problem, child[0], child[1])
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    break
            if Config.METRICS in Config.METRICS_MAX:
                if fit > pop[i][self.ID_FIT]:
                    pop[i] = [child, fit]
            else:
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [child, fit]

        if fe_mode is None:
            return pop
        else:
            counter = 2 * self.pop_size  # pop_new + pop_mutation operations
            return pop, counter
