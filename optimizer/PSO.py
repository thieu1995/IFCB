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


class BasePSO(Root):
    ID_POS = 0              # Current position
    ID_FIT = 1              # Current fitness
    ID_VEL = 2              # Current velocity
    ID_LOCAL_POS = 3        # Personal best location
    ID_LOCAL_FIT = 4        # Personal best fitness

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, time_bound=None, domain_range=None, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, time_bound, domain_range)
        if paras is None:
            paras = {"w_min": 0.4, "w_max": 0.9, "c_local": 1.2, "c_global": 1.2}
        self.w_min = paras["w_min"]
        self.w_max = paras["w_max"]
        self.c_local = paras["c_local"]
        self.c_global = paras["c_global"]

    def create_solution(self):
        while True:
            pos_mt_cloud = uniform(self.domain_range[0], self.domain_range[1], (len(self.problem["tasks"]), len(self.problem["clouds"])))
            pos_mt_fog = uniform(self.domain_range[0], self.domain_range[1], (len(self.problem["tasks"]), len(self.problem["fogs"])))
            schedule = matrix_to_schedule(self.problem, [pos_mt_cloud, pos_mt_fog])
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                vel_mt_cloud = uniform(self.domain_range[0], self.domain_range[1], (len(self.problem["tasks"]), len(self.problem["clouds"])))
                vel_mt_fog = uniform(self.domain_range[0], self.domain_range[1], (len(self.problem["tasks"]), len(self.problem["fogs"])))
                break
        return [[pos_mt_cloud, pos_mt_fog], fitness, [vel_mt_cloud, vel_mt_fog],
                [deepcopy(pos_mt_cloud), deepcopy(pos_mt_fog)], deepcopy(fitness)]
        # [solution, fit, velocity, local_solution, local_fitness]

    def evolve(self, pop, fe_mode=None, epoch=None, g_best=None):
        # Update weight after each move count  (weight down)
        w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min

        for i in range(self.pop_size):
            while True:
                x_new = []
                v_new = []
                for j in range(len(pop[i][self.ID_POS])):
                    v_temp = w * pop[i][self.ID_VEL][j] + self.c_local * uniform() * (pop[i][self.ID_LOCAL_POS][j] - pop[i][self.ID_POS][j]) + \
                        self.c_global * uniform() * (g_best[self.ID_POS][j] - pop[i][self.ID_POS][j])
                    x_temp = pop[i][self.ID_POS][j] + v_temp         # Xi(new) = Xi(old) + Vi(new) * deltaT (deltaT = 1)
                    x_temp = self.amend_position_random(x_temp)
                    x_new.append(x_temp)
                    v_new.append(v_temp)
                schedule = matrix_to_schedule(self.problem, x_new)
                if schedule.is_valid():
                    fit_new = self.Fit.fitness(schedule)
                    pop[i][self.ID_POS] = x_new
                    pop[i][self.ID_FIT] = fit_new
                    pop[i][self.ID_VEL] = v_new
                    # Update current position, current velocity and compare with past position, past fitness (local best)
                    if Config.METRICS in Config.METRICS_MAX:
                        if fit_new > pop[i][self.ID_LOCAL_FIT]:
                            pop[i][self.ID_LOCAL_POS] = x_new
                            pop[i][self.ID_LOCAL_FIT] = fit_new
                    else:
                        if fit_new < pop[i][self.ID_LOCAL_FIT]:
                            pop[i][self.ID_LOCAL_POS] = x_new
                            pop[i][self.ID_LOCAL_FIT] = fit_new
                    break

        if fe_mode is None:
            return pop
        else:
            counter = 2 * self.pop_size  # pop_new + pop_mutation operations
            return pop, counter
