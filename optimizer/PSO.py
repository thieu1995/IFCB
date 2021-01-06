#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:32, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import time

import numpy as np

from algorithms.fitness_manager import FitnessManager
from config import Config
from utils import get_min_value, matrix_to_schedule


class Particle:
    def __init__(self, cloud_matrix=None, fog_matrix=None, clouds=None, fogs=None, tasks=None,
                 fog_cloud_paths=None, element_value_range=None, trade_off_case=None):
        self.cloud_matrix = cloud_matrix
        self.fog_matrix = fog_matrix
        self.clouds = clouds
        self.fogs = fogs
        self.tasks = tasks
        self.time_scheduling = len(self.tasks) / 100 * 60
        self.fog_cloud_paths = fog_cloud_paths

        self.pbest_cloud_matrix = self.cloud_matrix
        self.pbest_fog_matrix = self.fog_matrix

        self.velocity_cloud_matrix = np.zeros(self.cloud_matrix.shape)
        self.velocity_fog_matrix = np.zeros(self.fog_matrix.shape)

        if Config.METRICS == 'trade-off':
            self.pbest_value = float('-inf')
        else:
            self.pbest_value = float('inf')

        self.schedule = matrix_to_schedule(self.cloud_matrix, self.fog_matrix, self.fog_cloud_paths)
        self.fitness_manager = FitnessManager(self.clouds, self.fogs, self.tasks)
        if Config.METRICS == 'trade-off':
            self.min_value_information = get_min_value(element_value_range)
            self.fitness_manager.set_min_power(self.min_value_information[str(len(self.tasks))]['power'])
            self.fitness_manager.set_min_latency(self.min_value_information[str(len(self.tasks))]['latency'])
            self.fitness_manager.set_min_cost(self.min_value_information[str(len(self.tasks))]['cost'])
            self.fitness_manager.set_trade_off(trade_off_case)

    def fitness(self):
        self.schedule = matrix_to_schedule(self.cloud_matrix, self.fog_matrix, self.fog_cloud_paths)
        fitness = self.fitness_manager.calc(self.schedule)
        return fitness


class PSOEngine:

    def __init__(self, population_size=10, epochs=200):
        self.population_size = population_size
        self.epochs = epochs

        self.particles = []
        if Config.METRICS == 'trade-off':
            self.gbest_value = float('-inf')
        else:
            self.gbest_value = float('inf')
        self.gbest_cloud_matrix = None
        self.gbest_fog_matrix = None
        self.gbest_particle = None

        self.max_w_old_velocation = 0.9
        self.min_w_old_velocation = 0.1
        self.w_local_best_position = 1.5
        self.w_global_best_position = 1.5

    def set_gbest(self):
        if Config.METRICS == 'trade-off':
            for particle in self.particles:
                fitness_candidate = particle.fitness()
                if particle.pbest_value < fitness_candidate:
                    particle.pbest_value = fitness_candidate
                    particle.pbest_cloud_matrix = particle.cloud_matrix
                    particle.pbest_fog_matrix = particle.fog_matrix

                if self.gbest_value < fitness_candidate:
                    self.gbest_value = fitness_candidate
                    self.gbest_cloud_matrix = particle.cloud_matrix
                    self.gbest_fog_matrix = particle.fog_matrix
                    self.gbest_particle = particle
        else:
            for particle in self.particles:
                fitness_candidate = particle.fitness()
                if particle.pbest_value > fitness_candidate:
                    particle.pbest_value = fitness_candidate
                    particle.pbest_cloud_matrix = particle.cloud_matrix
                    particle.pbest_fog_matrix = particle.fog_matrix

                if self.gbest_value > fitness_candidate:
                    self.gbest_value = fitness_candidate
                    self.gbest_cloud_matrix = particle.cloud_matrix
                    self.gbest_fog_matrix = particle.fog_matrix
                    self.gbest_particle = particle

    def move_particles(self):
        for particle in self.particles:
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()

            change_cloud_base_on_old_velocity = self.w_old_velocation * particle.velocity_cloud_matrix
            change_fog_base_on_old_velocity = self.w_old_velocation * particle.velocity_fog_matrix

            change_cloud_base_on_local_best = \
                self.w_local_best_position * r1 * (particle.pbest_cloud_matrix - particle.cloud_matrix)
            change_fog_base_on_local_best = \
                self.w_local_best_position * r1 * (particle.pbest_fog_matrix - particle.fog_matrix)

            change_cloud_base_on_global_best = \
                self.w_global_best_position * r2 * (self.gbest_cloud_matrix - particle.cloud_matrix)
            change_fog_base_on_global_best = \
                self.w_global_best_position * r2 * (self.gbest_fog_matrix - particle.fog_matrix)

            new_velocity_cloud_matrix = \
                change_cloud_base_on_old_velocity + change_cloud_base_on_local_best + change_cloud_base_on_global_best
            new_velocity_fog_matrix = \
                change_fog_base_on_old_velocity + change_fog_base_on_local_best + change_fog_base_on_global_best

            particle.velocity_cloud_matrix = new_velocity_cloud_matrix
            particle.velocity_fog_matrix = new_velocity_fog_matrix

            # print(f'particle cloud matrix before: {particle.cloud_matrix}')
            particle.cloud_matrix = particle.cloud_matrix + particle.velocity_cloud_matrix
            # print(f'particle cloud matrix after: {particle.cloud_matrix}')
            # print('------------------------------')
            particle.fog_matrix = particle.fog_matrix + particle.velocity_fog_matrix

    def early_stopping(self, array, patience=20):
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

    def check_most_n_value(self, fitness_arr, n):
        check = 0
        for i in range(len(fitness_arr) - 2, len(fitness_arr) - n, -1):
            if fitness_arr[i] == fitness_arr[-1]:
                check += 1
            if check == 4:
                return True
        return False

    def evolve(self):
        if Config.MODE == 'epochs':
            print('|-> Start tuning by particle swarm optimization')
            fitness_arr = []
            for iteration in range(self.epochs):
                self.w_old_velocation = \
                    (self.epochs - iteration) / self.epochs * (self.max_w_old_velocation - self.min_w_old_velocation) \
                    + self.min_w_old_velocation
                start_time = time.time()
                self.set_gbest()
                self.move_particles()
                # print('===> self.gbest_particle.position: {}'.format(self.gbest_particle.position))
                fitness_arr.append(round(self.gbest_value, 8))
                print(f'iteration: {iteration} fitness = {self.gbest_value:.8f}'
                      f' with time for running: {time.time() - start_time:.2f}')
                if iteration % 100 == 0:
                    print(fitness_arr)
            print(f'iterations: {iteration}: best fitness = {self.gbest_value}')
            return self.gbest_cloud_matrix, self.gbest_fog_matrix, np.array(fitness_arr)
        else:
            print('|-> Start tuning by particle swarm optimization')
            start_time_run = time.time()
            fitness_arr = []
            for iteration in range(self.epochs):
                self.w_old_velocation = \
                    (self.epochs - iteration) / self.epochs * (self.max_w_old_velocation - self.min_w_old_velocation) \
                    + self.min_w_old_velocation
                start_time = time.time()
                self.set_gbest()
                self.move_particles()
                # print('===> self.gbest_particle.position: {}'.format(self.gbest_particle.position))
                fitness_arr.append(round(self.gbest_value, 8))
                print(f'iteration: {iteration} fitness = {self.gbest_value:.8f}'
                      f' with time for running: {time.time() - start_time:.2f}')
                if iteration % 100 == 0:
                    print(fitness_arr)
                if time.time() - start_time_run >= self.gbest_particle.time_scheduling:
                    print('=== over time training ===')
                    break
            print(f'iterations: {iteration}: best fitness = {self.gbest_value}')
            return self.gbest_cloud_matrix, self.gbest_fog_matrix, np.array(fitness_arr)
