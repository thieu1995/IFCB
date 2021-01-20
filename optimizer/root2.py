#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:35, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from time import time
from config import Config
from sys import exit
from optimizer.root import Root
from numpy import array, inf, zeros, argmin
from numpy.random import uniform
from utils.schedule_util import matrix_to_schedule
from uuid import uuid4
from copy import deepcopy
from math import sqrt
from random import randint


class Root2(Root):
    ID_IDX = 0
    ID_POS = 1
    ID_FIT = 2
    

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub)
        self.num_obj = 3

    def create_solution(self):
        while True:
            matrix = uniform(self.lb, self.ub, self.problem["shape"])
            schedule = matrix_to_schedule(self.problem, matrix)
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                idx = uuid4().hex
                break
        return [idx, matrix, fitness]

    # Function to sort by values
    def sort_by_values(self, front: list, obj_list: array):
        sorted_list = []
        while (len(sorted_list) != len(front)):
            idx_min = argmin(obj_list)
            if idx_min in front:
                sorted_list.append(idx_min)
            obj_list[idx_min] = inf
        return sorted_list

    # Function to calculate crowding distance
    def crowding_distance(self, pop: dict, front: list):
        obj1, obj2, obj3 = zeros(len(pop)), zeros(len(pop)), zeros(len(pop))
        for idx, item in enumerate(pop.values()):
            obj1[idx] = item[self.ID_FIT][0]
            obj2[idx] = item[self.ID_FIT][1]
            obj3[idx] = item[self.ID_FIT][2]
        distance = [0.0 for _ in range(0, len(front))]
        sorted1 = self.sort_by_values(front, obj1)
        sorted2 = self.sort_by_values(front, obj2)
        sorted3 = self.sort_by_values(front, obj3)
        distance[0] = 1e10
        distance[len(front) - 1] = 1e10
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (obj1[sorted1[k + 1]] - obj1[sorted1[k - 1]]) / (max(obj1) - min(obj1) + 1e-5)
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (obj2[sorted2[k + 1]] - obj2[sorted2[k - 1]]) / (max(obj2) - min(obj2) + 1e-5)
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (obj3[sorted3[k + 1]] - obj3[sorted3[k - 1]]) / (max(obj3) - min(obj3) + 1e-5)
        return distance
    
    def dominate(self, id1, id2, obj):
        better = False
        for i in range(self.num_obj):
            if obj[i][id1] > obj[i][id2]:
                return False
            elif obj[i][id1] < obj[i][id2]:
                better = True
        return better
    
    # Function to carry out NSGA-II's fast non dominated sort
    def fast_non_dominated_sort(self, pop: dict):
        obj = [[] for _ in range(0, self.num_obj)]
        for idx, item in pop.items():
            for i in range(self.num_obj):
                obj[i].append(item[self.ID_FIT][i])
        size = len(obj[0])
        front = []
        num_assigned_individuals = 0
        indv_ranks = [0 for _ in range(0, size)]
        rank = 1
        
        while num_assigned_individuals < self.pop_size:
            cur_front = []
            for i in range(self.pop_size):
                if indv_ranks[i] > 0:
                    continue
                be_dominated = False
                
                j = 0
                while j < len(cur_front):
                    idx_1 = cur_front[j]
                    idx_2 = i
                    if self.dominate(idx_1, idx_2, obj):
                        be_dominated = True
                        break
                    elif self.dominate(idx_2, idx_1, obj):
                        cur_front[j] = cur_front[-1]
                        cur_front.pop()
                        j -= 1
                    j += 1
                        
                if(not be_dominated):
                    cur_front.append(i)
                    
            for i in range(len(cur_front)):
                indv_ranks[ cur_front[i] ] = rank
            front.append(cur_front)
            num_assigned_individuals += len(cur_front)
            rank += 1
        return front

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        pass

    def train(self):
        if Config.METRICS == "pareto" and self.__class__.__name__ not in Config.MULTI_OBJECTIVE_SUPPORTERS:
            print(f'Method: {self.__class__.__name__} doesn"t support pareto-front fitness function type')
            exit()
        print(f'Start training by: {self.__class__.__name__} algorithm, fitness type: {Config.METRICS}')

        pop_temp = [self.create_solution() for _ in range(self.pop_size)]
        pop = {item[self.ID_IDX]: item for item in pop_temp}

        time_bound_start = time()
        time_bound_log = "without time-bound."
        if Config.TIME_BOUND_KEY:
            time_bound_log = f'with time-bound: {Config.TIME_BOUND_VALUE} seconds.'
        if Config.MODE == 'epoch':
            print(f'Training by: epoch (mode) with: {self.epoch} epochs, {time_bound_log}')
            g_best_dict = {}
            for epoch in range(self.epoch):
                time_epoch_start = time()
                pop = self.evolve(pop, None, epoch, None)
                front = self.fast_non_dominated_sort(pop)
                current_best = []
                front0 = front[0]
                for it in front0:
                    current_best.append(list(pop.values())[it][self.ID_FIT])
                g_best_dict[epoch] = array(current_best)
                time_epoch_end = time() - time_epoch_start
                print(f'Front size: {len(front[0])}, including {list(pop.values())[front[0][0]][self.ID_FIT]}, '
                      f'Epoch {epoch + 1} with time: {time_epoch_end:.2f} seconds')
                if Config.TIME_BOUND_KEY:
                    if time() - time_bound_start >= Config.TIME_BOUND_VALUE:
                        print('====== Over time for training ======')
                        break
            solutions = {}
            g_best = []
            front0 = front[0]
            for it in front0:
                idx = list(pop.keys())[it]
                solutions[idx] = pop[idx]
                g_best.append(pop[idx][self.ID_FIT])
            return solutions, array(g_best), g_best_dict

