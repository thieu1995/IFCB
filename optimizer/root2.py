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


class Root2(Root):
    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub)

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
            distance[k] = distance[k] + (obj1[sorted1[k + 1]] - obj1[sorted1[k - 1]]) / (max(obj1) - min(obj1))
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (obj2[sorted2[k + 1]] - obj2[sorted2[k - 1]]) / (max(obj2) - min(obj2))
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (obj3[sorted3[k + 1]] - obj3[sorted3[k - 1]]) / (max(obj3) - min(obj3))
        return distance

    # Function to carry out NSGA-II's fast non dominated sort
    def fast_non_dominated_sort(self, pop: dict):
        obj1, obj2, obj3 = [], [], []
        for idx, item in pop.items():
            obj1.append(item[self.ID_FIT][0])
            obj2.append(item[self.ID_FIT][1])
            obj3.append(item[self.ID_FIT][2])
        size = len(obj1)
        S = [[] for _ in range(0, size)]
        front = [[]]
        n = [0 for _ in range(0, size)]
        rank = [0 for _ in range(0, size)]

        for p in range(0, size):
            S[p] = []
            n[p] = 0
            for q in range(0, size):
                if (obj1[p] > obj1[q] and obj2[p] > obj2[q] and obj3[p] > obj3[q]) \
                        or (obj1[p] >= obj1[q] and obj2[p] > obj2[q] and obj3[p] > obj3[q]) \
                        or (obj1[p] > obj1[q] and obj2[p] >= obj2[q] and obj3[p] > obj3[q]) \
                        or (obj1[p] > obj1[q] and obj2[p] > obj2[q] and obj3[p] >= obj3[q]):
                    if q not in S[p]:
                        S[p].append(q)
                elif (obj1[q] > obj1[p] and obj2[q] > obj2[p] and obj3[q] > obj3[p]) \
                        or (obj1[q] >= obj1[p] and obj2[q] > obj2[p] and obj3[q] > obj3[p]) \
                        or (obj1[q] > obj1[p] and obj2[q] >= obj2[p] and obj3[q] > obj3[p]) \
                        or (obj1[q] > obj1[p] and obj2[q] > obj2[p] and obj3[q] >= obj3[p]):
                    n[p] = n[p] + 1
            if n[p] == 0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)
        i = 0
        while (front[i] != []):
            Q = []
            for p in front[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if (n[q] == 0):
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)
            i = i + 1
            front.append(Q)
        del front[len(front) - 1]
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

