#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:12, 27/04/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%
from time import time

from config import Config
from optimizer.root3 import Root3
from numpy.random import uniform, randint, normal, choice
from numpy import sum, exp, ones, where, sqrt, array
from utils.schedule_util import matrix_to_schedule
from uuid import uuid4


class BaseMO_SSA_OLD(Root3):
    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        if paras is None:
            paras = {"ST": 0.8, "PD": 0.2, "SD": 0.1}
        self.ST = paras["ST"]  # ST in [0.5, 1.0]
        self.PD = paras["PD"]  # number of producers                        --> In MO: this is the first-front
        self.SD = paras["SD"]  # number of sparrows who perceive the danger --> In MO: this is the last-front

    def sort_population(self, pop: list):
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
        n1_invert = sum(dominated_list)  ## number of dominated
        n1 = int(self.pop_size - n1_invert)  ## number of non-dominated -> producers
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

    def train(self):
        time_counting = time()
        if Config.METRICS == "pareto" and self.__class__.__name__ not in Config.MULTI_OBJECTIVE_SUPPORTERS:
            print(f'Method: {self.__class__.__name__} doesn"t support pareto-front fitness function type')
            exit()
        if self.verbose:
            print(f'Start training by: {self.__class__.__name__} algorithm, fitness type: {Config.METRICS}')

        pop = [self.create_solution() for _ in range(self.pop_size)]
        self.n_objs = len(pop[0][self.ID_FIT])
        # The archive population (at first: its size is the same size with the population, then its changing iteratively)
        pop_archive = self.create_opposition_based_pop(pop)
        g_best_dict = {}  # Fronts[0] by iterations
        # The best solution which is an archive member and it in the least populated area as an attractant to improve coverage
        global_best = self.find_global_best(pop)  # Helping AntLion only, the results we want is the first-pareto-front

        training_info = {"Epoch": [], "FrontSize": [], "ArchiveSize": [], "Time": []}
        time_bound_start = time()
        time_bound_log = f'with time-bound: {Config.TIME_BOUND_VALUE} seconds.' if Config.TIME_BOUND_KEY else "without time-bound."
        if Config.MODE == 'epoch':
            if self.verbose:
                print(f'Training by: epoch (mode) with: {self.epoch} epochs, {time_bound_log}')
            for epoch in range(self.epoch):
                time_epoch_start = time()

                pop = self.evolve2(pop, pop_archive, None, epoch, global_best)

                pop_new = pop_archive + pop
                pop_archive, archive_fits = self.update_pop_archive(pop_new)

                pop_new = pop_archive + pop
                non_dominated_list = self.find_dominates_list(pop_new)
                current_best = []
                for idx, value in enumerate(non_dominated_list):
                    if value == 0:
                        current_best.append(pop_new[idx][self.ID_FIT])
                g_best_dict[epoch + 1] = array(current_best)

                time_epoch_end = time() - time_epoch_start
                training_info = self.adding_element_to_dict(training_info, ["Epoch", "FrontSize", "ArchiveSize", "Time"],
                                                            [epoch + 1, len(non_dominated_list), len(pop_archive), time_epoch_end])
                if self.verbose:
                    print(f'Epoch: {epoch + 1}, Front0 size: {len(non_dominated_list)}, Archive size: {len(pop_archive)}, '
                          f'First front0 fit: {g_best_dict[epoch][0]}, with time: {time_epoch_end:.2f} seconds')
                if Config.TIME_BOUND_KEY:
                    if time() - time_bound_start >= Config.TIME_BOUND_VALUE:
                        print('====== Over time for training ======')
                        break
            solutions = {}
            g_best = []
            for idx, value in enumerate(non_dominated_list):
                if value == 0:
                    g_best.append(pop_new[idx][self.ID_FIT])
            time_counting = time() - time_counting
            training_info = self.adding_element_to_dict(training_info, ["Epoch", "FrontSize", "ArchiveSize", "Time"],
                                                        [epoch + 1, len(non_dominated_list), len(pop_archive), time_counting])
            return solutions, array(g_best), g_best_dict, training_info

