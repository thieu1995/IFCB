#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 08:52, 01/02/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from optimizer.root3 import Root3
from numpy.random import uniform, randint, normal, choice
from numpy import min, max, cumsum, reshape, sum, exp, ones, where, sqrt
from utils.schedule_util import matrix_to_schedule
from uuid import uuid4


class BaseMO_SSA(Root3):
    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        if paras is None:
            paras = {"ST": 0.8, "PD": 0.2, "SD": 0.1}
        self.ST = paras["ST"]  # ST in [0.5, 1.0]
        self.PD = paras["PD"]  # number of producers                        --> In MO: this is the first-front
        self.SD = paras["SD"]  # number of sparrows who perceive the danger --> In MO: this is the last-front

    def sort_population(self, pop:list):
        pop_first, pop_last = [], []
        size = len(pop)
        dominated_list = self.find_dominates_list(pop)
        for i in range(0, size):
            if dominated_list[i] == 0:
                pop_first.append(pop[i])
            else:
                pop_last.append(pop[i])
        return pop_first + pop_last

    def evolve2(self, pop: list, pop_archive: list, fe_mode=None, epoch=None, g_best=None):
        # Updating population by SSA-equations.
        pop = self.sort_population(pop)
        global_best_pos = g_best[self.ID_POS]
        random_solution = pop_archive[randint(0, len(pop_archive))]

        dominated_list = self.find_dominates_list(pop)
        n1_invert = sum(dominated_list)             ## number of dominated
        n1 = int(self.pop_size - n1_invert)         ## number of non-dominated -> producers
        n2 = int(self.SD * self.pop_size)

        # Using equation (3) update the sparrow’s location;
        for i in range(0, n1):
            while True:
                if uniform() < self.ST:  # R2 in [0, 1], the alarm value, random value
                    x_new = pop[i][self.ID_POS] * exp((epoch + 1) / self.epoch)
                else:
                    x_new = pop[i][self.ID_POS] + normal() * ones(self.problem["shape"])
                child = self.amend_position_random(x_new)
                schedule = matrix_to_schedule(self.problem, child)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    idx = uuid4().hex
                    break
            pop[i] = [idx, child, fit]

        # Using equation (4) update the sparrow’s location;
        shape = g_best[self.ID_POS].shape
        for i in range(n1, self.pop_size):
            while True:
                if i > int(self.pop_size / 2):
                    x_new = normal() * exp((random_solution[self.ID_POS] - pop[i][self.ID_POS]) / ((i + 1) ** 2))
                else:
                    # A = sign(uniform(-1, 1, (1, shape[0]*shape[1])))
                    # A1 = matmul(A.T, inv(matmul(A, A.T)))
                    # A1 = matmul(A1, ones((1, shape[0]*shape[1])))
                    # temp = reshape(abs(pop[i][self.ID_POS] - global_best_pos), (shape[0] * shape[1]))
                    # temp = matmul(temp, A1)
                    # x_new = global_best_pos + uniform() * reshape(temp, shape)
                    x_new = global_best_pos + normal(0, 1, shape) * abs(pop[i][self.ID_POS] - global_best_pos)
                child = self.amend_position_random(x_new)
                schedule = matrix_to_schedule(self.problem, child)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    idx = uuid4().hex
                    break
            pop[i] = [idx, child, fit]

        ## Using equation (5) update the sparrow’s location;
        n2_list = choice(list(range(0, self.pop_size)), n2, replace=False)
        dominated_list = self.find_dominates_list(pop)
        non_dominated_list = where(dominated_list == 0)[0]
        for i in n2_list:
            if i in non_dominated_list:
                x_new = global_best_pos + normal() * abs(pop[i][self.ID_POS] - global_best_pos)
            else:
                dist = sum(sqrt((pop[i][self.ID_FIT] - g_best[self.ID_FIT]) ** 2))
                x_new = pop[i][self.ID_POS] + uniform(-1, 1) * (abs(pop[i][self.ID_POS] - global_best_pos) / (dist + self.EPSILON))
            while True:
                child = self.amend_position_random(x_new)
                schedule = matrix_to_schedule(self.problem, child)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    idx = uuid4().hex
                    break
            pop[i] = [idx, child, fit]
        return pop