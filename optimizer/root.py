#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:35, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import time
import numpy as np
from config import Config
from model import Fitness
from utils import get_min_value, matrix_to_schedule
from utils import create_solution


class Root:
    def __init__(self, pop_size=10, epoch=2, func_eval=100000, time_bound=None, domain_range=None):
        self.pop_size = pop_size
        self.epoch = epoch
        self.func_eval = func_eval
        self.time_bound = time_bound
        self.domain_range = domain_range

    def create_solution(self):
        sol, fit = create_solution(self.domain_range)
        return [sol, fit]

    def cal_rank(self, pop):
        '''
        Calculate ranking for element in current population
        '''
        fit = []
        for i in range(len(pop)):
            fit.append(pop[i][1])
        arg_rank = np.array(fit).argsort()
        rank = [i / sum(range(1, len(pop) + 1)) for i in range(1, len(pop) + 1)]
        return rank

    def wheel_select(self, pop, prob):
        '''
        Select dad and mom from current population by rank
        '''
        r = np.random.random()
        sum = prob[0]
        for i in range(1, len(pop) + 1):
            if sum > r:
                return i - 1
            else:
                sum += prob[i]
        return sum

    def early_stopping(self, array, patience=5):
        if patience <= len(array) - 1:
            value = array[len(array) - patience]
            arr = array[len(array) - patience + 1:]
            check = 0
            for val in arr:
                if val < value:
                    check += 1
            if check != 0:
                return False
            return True
        raise ValueError

    def evolve(self):
        print('|-> Start evolve with genertic algorithms')

        pop = [self.create_solution() for _ in range(self.population_size)]

        if Config.METRICS == 'trade-off':
            gbest = max(pop, key=lambda x: x[1])
        else:
            gbest = min(pop, key=lambda x: x[1])

        g_best_arr = [gbest[1]]
        if Config.MODE == 'epochs':
            for iter in range(self.epochs):
                print('Iteration {}'.format(iter + 1))
                start_time = time.time()
                pop = self.select(pop)
                pop = self.mutate(pop)
                if Config.METRICS == 'trade-off':
                    best_fit = max(pop, key=lambda x: x[1])
                    if best_fit[1] > g_best_arr[-1]:
                        gbest = best_fit
                else:
                    best_fit = min(pop, key=lambda x: x[1])
                    if best_fit[1] < g_best_arr[-1]:
                        gbest = best_fit
                g_best_arr.append(gbest[1])
                time_run = time.time() - start_time
                print(f'best current fit {best_fit[1]:.8f}, '
                      f'best fit so far {gbest[1]:.8f}, '
                      f'iter {iter} with time: {time_run:.2f}')
            return gbest[0][0], gbest[0][1], np.array(g_best_arr)
        else:
            print('==== oprimize by time ===')
            print('time scheduling: ', self.time_scheduling)
            start_time_run = time.time()
            for iter in range(self.epochs):
                print('Iteration {}'.format(iter + 1))
                start_time_epoch = time.time()
                pop = self.select(pop)
                pop = self.mutate(pop)
                if Config.METRICS == 'trade-off':
                    best_fit = max(pop, key=lambda x: x[1])
                    if best_fit[1] > g_best_arr[-1]:
                        gbest = best_fit
                else:
                    best_fit = min(pop, key=lambda x: x[1])
                    if best_fit[1] < g_best_arr[-1]:
                        gbest = best_fit
                g_best_arr.append(gbest[1])
                time_run = time.time() - start_time_epoch
                print(f'best current fit {best_fit[1]:.8f}, '
                      f'best fit so far {gbest[1]:.8f}, '
                      f'iter {iter} with time: {time_run:.2f}')
                if time.time() - start_time_run >= self.time_scheduling:
                    print('=== over time training ===')
                    break
            return gbest[0][0], gbest[0][1], np.array(g_best_arr)

