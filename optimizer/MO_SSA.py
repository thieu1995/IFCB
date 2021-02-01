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
from numpy import min, max, cumsum, reshape, sum, exp, ones, where, sign, sqrt, matmul
from numpy.linalg import inv
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

    def evolve2(self, pop: list, pop_archive: list, fe_mode=None, epoch=None, g_best=None):
        # Updating population by SSA-equations.
        global_best_pos = g_best[self.ID_POS]
        random_solution = pop_archive[randint(0, len(pop_archive))][self.ID_POS].flatten()

        dominated_list = self.find_dominates_list(pop)
        n1_invert = sum(dominated_list)             ## number of dominated
        n1 = int(self.pop_size - n1_invert)         ## number of non-dominated -> current best
        n2 = int(self.SD * self.pop_size)
        if n1 < 5 or n1 > self.pop_size - 5:
            n1 = int(self.PD * self.pop_size)
        ## Find current worst
        worst_list = where(dominated_list == 1)[0]
        worst_idx = choice(list(range(0, self.pop_size)))
        if len(worst_list) != 0:
            worst_idx = choice(worst_list)
        current_worst = pop[worst_idx]

        r2 = uniform()  # R2 in [0, 1], the alarm value, random value
        # Using equation (3) update the sparrow’s location;
        for i in range(0, n1):
            if r2 < self.ST:
                x_new = pop[i][self.ID_POS] * exp((i + 1) / (uniform(self.EPSILON, 1) * self.epoch))
            else:
                x_new = pop[i][self.ID_POS] + normal() * ones(self.problem["shape"])
            while True:
                child = self.amend_position_random(uniform() * x_new)
                schedule = matrix_to_schedule(self.problem, child)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    idx = uuid4().hex
                    break
            pop[i] = [idx, child, fit]

        ## Find current best
        pop_archive, archive_fits = self.update_pop_archive(pop_archive, pop[:n1])
        archive_ranks = self.neighbourhood_ranking(archive_fits)
        idx = self.roulette_wheel_selection(1.0 / archive_ranks)
        idx = choice(list(range(0, len(pop_archive)))) if idx == -1 else idx
        current_best = pop_archive[idx]

        # Using equation (4) update the sparrow’s location;
        shape = current_best[self.ID_POS].shape
        for i in range(n1, self.pop_size):

            if i > int(self.pop_size / 2):
                x_new = normal() * exp((current_worst[self.ID_POS] - pop[i][self.ID_POS]) / (i + 1) ** 2)
            else:
                A = sign(uniform(-1, 1, (1, shape[0]*shape[1])))
                A1 = matmul(A.T, inv(matmul(A, A.T)))
                A1 = matmul(A1, ones((1, shape[0]*shape[1])))
                temp = reshape(abs(pop[i][self.ID_POS] - current_best[self.ID_POS]), (shape[0] * shape[1]))
                temp = matmul(temp, A1)
                x_new = current_best[self.ID_POS] + reshape(temp, shape)
            while True:
                child = self.amend_position_random(uniform() * x_new)
                schedule = matrix_to_schedule(self.problem, child)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    idx = uuid4().hex
                    break
            pop[i] = [idx, x_new, fit]

        #  Using equation (5) update the sparrow’s location;
        n2_list = choice(list(range(0, self.pop_size)), n2, replace=False)
        dominated_list = self.find_dominates_list(pop)
        non_dominated_list = where(dominated_list == 0)[0]
        for i in n2_list:
            if i in non_dominated_list:
                x_new = global_best_pos + normal() * abs(pop[i][self.ID_POS] - global_best_pos)
            else:
                dist = sum(sqrt((pop[i][self.ID_FIT] - current_worst[self.ID_FIT]) ** 2))
                x_new = pop[i][self.ID_POS] + uniform(-1, 1) * (abs(pop[i][self.ID_POS] - global_best_pos) / (dist + self.EPSILON))
            while True:
                child = self.amend_position_random(uniform()* x_new)
                schedule = matrix_to_schedule(self.problem, child)
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    idx = uuid4().hex
                    break
            pop[i] = [idx, x_new, fit]
        return pop