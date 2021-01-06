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

from config import *


class WoaEngine:
    def __init__(self, population_size=100, epochs=500):
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

    def evolve(self):
        print('start with woa')
        if Config.MODE == 'epochs':
            fitness_arr = []
            for epoch in range(self.epochs):
                epoch_start_time = time.time()
                self.set_gbest()
                a = 2 * np.cos(epoch / (self.epochs - 1))

                for particle in self.particles:

                    r = np.random.rand()
                    A = 2 * a * r - a
                    C = 2 * r
                    l = np.random.uniform(-1, 1)
                    b = 1
                    p = np.random.rand()

                    if p < 0.5:
                        if np.abs(A) < 1:
                            D_cloud_matrix = np.abs(C * self.gbest_cloud_matrix - particle.cloud_matrix)
                            D_fog_matrix = np.abs(C * self.gbest_fog_matrix - particle.fog_matrix)

                            particle.cloud_matrix = self.gbest_cloud_matrix - A * D_cloud_matrix
                            particle.fog_matrix = self.gbest_fog_matrix - A * D_fog_matrix
                        else:
                            random_agent_idx = np.random.randint(0, len(self.particles))
                            random_particle = self.particles[random_agent_idx]
                            D_cloud_matrix = np.abs(C * random_particle.cloud_matrix - particle.cloud_matrix)
                            D_fog_matrix = np.abs(C * random_particle.fog_matrix - particle.fog_matrix)

                            particle.cloud_matrix = random_particle.cloud_matrix - A * D_cloud_matrix
                            particle.fog_matrix = random_particle.fog_matrix - A * D_fog_matrix
                    else:
                        D_cloud_matrix = np.abs(self.gbest_cloud_matrix - particle.cloud_matrix)
                        D_fog_matrix = np.abs(self.gbest_fog_matrix - particle.fog_matrix)

                        particle.cloud_matrix = D_cloud_matrix * np.exp(b * l) * np.cos(2 * np.pi * l) \
                                                + self.gbest_cloud_matrix

                        particle.fog_matrix = D_fog_matrix * np.exp(b * l) * np.cos(2 * np.pi * l) \
                                              + self.gbest_fog_matrix

                    # particle.fix_parameter_after_update()
                    # particle.move()
                fitness_arr.append(self.gbest_value)
                training_history = 'Iteration {}, best fitness = {} with time = {}' \
                    .format(epoch, fitness_arr[-1], round(time.time() - epoch_start_time, 4))
                print(training_history)
        else:
            fitness_arr = []
            start_time_run = time.time()
            for epoch in range(self.epochs):
                epoch_start_time = time.time()
                self.set_gbest()
                a = 2 * np.cos(epoch / (self.epochs - 1))

                for particle in self.particles:

                    r = np.random.rand()
                    A = 2 * a * r - a
                    C = 2 * r
                    l = np.random.uniform(-1, 1)
                    b = 1
                    p = np.random.rand()

                    if p < 0.5:
                        if np.abs(A) < 1:
                            D_cloud_matrix = np.abs(C * self.gbest_cloud_matrix - particle.cloud_matrix)
                            D_fog_matrix = np.abs(C * self.gbest_fog_matrix - particle.fog_matrix)

                            particle.cloud_matrix = self.gbest_cloud_matrix - A * D_cloud_matrix
                            particle.fog_matrix = self.gbest_fog_matrix - A * D_fog_matrix
                        else:
                            random_agent_idx = np.random.randint(0, len(self.particles))
                            random_particle = self.particles[random_agent_idx]
                            D_cloud_matrix = np.abs(C * random_particle.cloud_matrix - particle.cloud_matrix)
                            D_fog_matrix = np.abs(C * random_particle.fog_matrix - particle.fog_matrix)

                            particle.cloud_matrix = random_particle.cloud_matrix - A * D_cloud_matrix
                            particle.fog_matrix = random_particle.fog_matrix - A * D_fog_matrix
                    else:
                        D_cloud_matrix = np.abs(self.gbest_cloud_matrix - particle.cloud_matrix)
                        D_fog_matrix = np.abs(self.gbest_fog_matrix - particle.fog_matrix)

                        particle.cloud_matrix = D_cloud_matrix * np.exp(b * l) * np.cos(2 * np.pi * l) \
                                                + self.gbest_cloud_matrix

                        particle.fog_matrix = D_fog_matrix * np.exp(b * l) * np.cos(2 * np.pi * l) \
                                              + self.gbest_fog_matrix

                    # particle.fix_parameter_after_update()
                    # particle.move()
                fitness_arr.append(self.gbest_value)
                training_history = 'Iteration {}, best fitness = {} with time = {}' \
                    .format(epoch, fitness_arr[-1], round(time.time() - epoch_start_time, 4))
                print(training_history)
                if time.time() - start_time_run >= self.gbest_particle.time_scheduling:
                    print('=== over time training ===')
                    break
        return self.gbest_particle.cloud_matrix, self.gbest_particle.fog_matrix, np.array(fitness_arr)

