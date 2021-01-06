#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:32, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import random
import time

import numpy as np

from config import Config
from utils import get_min_value, matrix_to_schedule


class GAEngine:
    cross_over_rate = 0.9
    mutation_rate = 0.05

    def __init__(self, population_size=10, epochs=2, clouds=None, fogs=None, tasks=None, fog_cloud_paths=None,
                 element_value_range=None):
        self.population_size = population_size
        self.epochs = epochs
        self.clouds = clouds
        self.fogs = fogs
        self.tasks = tasks
        self.time_scheduling = len(self.tasks) / 100 * 60
        self.fog_cloud_paths = fog_cloud_paths
        self.element_value_range = element_value_range

        self.fitness_manager = FitnessManager(clouds, fogs, tasks)
        if Config.METRICS == 'trade-off':
            self.min_value_information = get_min_value(self.element_value_range)
            self.fitness_manager.set_min_power(self.min_value_information[str(len(self.tasks))]['power'])
            self.fitness_manager.set_min_latency(self.min_value_information[str(len(self.tasks))]['latency'])
            self.fitness_manager.set_min_cost(self.min_value_information[str(len(self.tasks))]['cost'])

    def compute_fitness(self, solution):
        return self.fitness_manager.calc(schedule=solution)

    def create_solution(self):
        is_not_valid = True
        while is_not_valid:
            cloud_matrix = np.random.uniform(
                self.element_value_range[0], self.element_value_range[1], (len(self.tasks), len(self.clouds)))
            fog_matrix = np.random.uniform(
                self.element_value_range[0], self.element_value_range[1], (len(self.tasks), len(self.fogs)))
            _schedule = matrix_to_schedule(cloud_matrix, fog_matrix, self.fog_cloud_paths)
            if _schedule.is_valid():
                is_not_valid = False
            _fitness = self.compute_fitness(_schedule)
        return [[cloud_matrix, fog_matrix], _fitness]

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

    def cross_over(self, dad_element, mom_element):
        '''
        crossover dad and mom choose from current population
        '''
        r = np.random.random()
        child_element = []
        if r < self.cross_over_rate:
            is_not_valid = True
            while is_not_valid:
                for i in range(len(dad_element[0])):
                    child_element.append((dad_element[0][i] + mom_element[0][i]) / 2)
                child_schedule = matrix_to_schedule(child_element[0], child_element[1], self.fog_cloud_paths)
                if child_schedule.is_valid():
                    is_not_valid = False
            return [child_element, self.compute_fitness(child_schedule)]
        if dad_element[1] < mom_element[1]:
            if Config.METRICS == 'trade-off':
                return mom_element
            else:
                return dad_element
        else:
            if Config.METRICS == 'trade-off':
                return dad_element
            else:
                return mom_element

    def select(self, pop):
        '''
        Select from current population and create new population
        '''
        new_pop = []
        sum_fit = 0
        for i in range(len(pop)):
            sum_fit += pop[0][1]
        while len(new_pop) < self.population_size:
            rank = self.cal_rank(pop)
            dad_index = self.wheel_select(pop, rank)
            mom_index = self.wheel_select(pop, rank)
            while dad_index == mom_index:
                mom_index = self.wheel_select(pop, rank)
            dad = pop[dad_index]
            mom = pop[mom_index]
            new_sol1 = self.cross_over(dad, mom)
            new_pop.append(new_sol1)
        return new_pop

    def mutate(self, pop):
        '''
        Mutate new population
        '''
        for i in range(len(pop)):
            is_not_valid = True
            while is_not_valid:
                for j in range(len(pop[i][0])):
                    if np.random.random() < self.mutation_rate:
                        num_value_change = random.choice(range(pop[i][0][j].shape[0] * pop[i][0][j].shape[1]))
                        for k in range(num_value_change):
                            task_idx = random.choice(range(pop[i][0][j].shape[0]))
                            device_idx = random.choice(range(pop[i][0][j].shape[1]))
                            pop[i][0][j][task_idx][device_idx] = random.uniform(-1, 1)
                schedule = matrix_to_schedule(pop[i][0][0], pop[i][0][1], self.fog_cloud_paths)
                if schedule.is_valid():
                    is_not_valid = False
            pop[i][1] = self.compute_fitness(schedule)
        return pop

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

