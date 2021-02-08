#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 10:44, 28/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from optimizer.root3 import Root3
from numpy.random import uniform, randint
from numpy import min, max, cumsum, reshape
from utils.schedule_util import matrix_to_schedule
from uuid import uuid4


class BaseMO_ALO(Root3):
    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)

    def random_walk_around_antlion(self, epoch_current, epoch_max, lb, ub, position, dim):
        I = 1  # I is the ratio in Equations (2.10) and (2.11)
        if epoch_current > epoch_max / 10:
            I = 1 + 100 * (epoch_current / epoch_max)
        if epoch_current > epoch_max / 2:
            I = 1 + 1000 * (epoch_current / epoch_max)
        if epoch_current > epoch_max * (3 / 4):
            I = 1 + 10000 * (epoch_current / epoch_max)
        if epoch_current > epoch_max * 0.9:
            I = 1 + 100000 * (epoch_current / epoch_max)
        if epoch_current > epoch_max * 0.95:
            I = 1 + 1000000 * (epoch_current / epoch_max)

        # Decrease boundaries to converge towards antlion
        lb, ub = lb / I, ub / I

        # Move the interval of [lb, ub] around the antlion [lb+anlion ub+antlion]
        lb += position if uniform() < 0.5 else (-position)  # Equation (2.8) in the paper
        ub += position if uniform() < 0.5 else (-position)  # Equation (2.9) in the paper

        # This function creates n random walks and normalize according to lb and ub vector
        X = cumsum(2 * (uniform(0, 1, (dim, epoch_max)) > 0.5) - 1, axis=1)  # Equation (2.1) in the paper, [a b]--->[c d]
        a = reshape(min(X, axis=1), (-1, 1))
        b = reshape(max(X, axis=1), (-1, 1))
        c = reshape(lb, (-1, 1))
        d = reshape(ub, (-1, 1))
        matrix_walk = (X - a) * ((d-c)/(b-a)) + c
        return matrix_walk.T

    def evolve2(self, pop:list, pop_archive:list, fe_mode=None, epoch=None, g_best=None):
        # Generating new population with double size using ALO equations
        #   random_solution = pop_archive[0]
        random_solution = pop_archive[randint(0, len(pop_archive))]
        for i in range(0, self.pop_size):
            random_antlion_pos = random_solution[self.ID_POS].flatten()
            dim = len(random_antlion_pos)
            global_best_pos = g_best[self.ID_POS].flatten()
            RA = self.random_walk_around_antlion(epoch, self.epoch, self.lb, self.ub, random_antlion_pos, dim)
            RE = self.random_walk_around_antlion(epoch, self.epoch, self.lb, self.ub, global_best_pos, dim)
            child = (RA[epoch] + RE[epoch]) / 2
            child = reshape(child, self.problem["shape"])
            while True:
                matrix = uniform() * child
                matrix = self.amend_position_random(matrix)
                schedule = matrix_to_schedule(self.problem, matrix)
                if schedule.is_valid():
                    fitness = self.Fit.fitness(schedule)
                    idx = uuid4().hex
                    break
            pop[i] = [idx, matrix, fitness]
        return pop