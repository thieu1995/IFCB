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
from numpy import array, inf, zeros, argmin, sqrt, hstack
from numpy import sum, where, ones, dot, power, subtract, multiply
from numpy import min as np_min
from numpy.random import uniform, random
from random import randint
from utils.schedule_util import matrix_to_schedule
from uuid import uuid4
from copy import deepcopy


class Root2(Root):
    ID_IDX = 0
    ID_POS = 1
    ID_FIT = 2

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub)
        self.n_objs = None
        self.WEIGHT = 1e-6

    def create_solution(self):
        while True:
            matrix = uniform(self.lb, self.ub, self.problem["shape"])
            schedule = matrix_to_schedule(self.problem, matrix)
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                idx = uuid4().hex
                break
        return [idx, matrix, fitness]

    def mutate(self, child, p_m):
        while True:
            child_pos = deepcopy(child[self.ID_POS])
            rd_matrix = uniform(self.lb, self.ub, self.problem["shape"])
            child_pos[rd_matrix < p_m] = 0
            rd_matrix_new = uniform(self.lb, self.ub, self.problem["shape"])
            rd_matrix_new[rd_matrix >= p_m] = 0
            child_pos = child_pos + rd_matrix_new
            schedule = matrix_to_schedule(self.problem, child_pos)
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                idx = uuid4().hex
                break
        return [idx, child_pos, fitness]

    def crossover(self, dad, mom, p_c):
        if random() < p_c:
            child = (dad[self.ID_POS] + mom[self.ID_POS]) / 2
            schedule = matrix_to_schedule(self.problem, child)
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                idx = uuid4().hex
            else:
                while True:
                    coef = uniform(0, 1)
                    child = coef * dad[self.ID_POS] + (1 - coef) * mom[self.ID_POS]
                    schedule = matrix_to_schedule(self.problem, child)
                    if schedule.is_valid():
                        fitness = self.Fit.fitness(schedule)
                        idx = uuid4().hex
                        break
            return [idx, child, fitness]
        return dad

    # Function to sort by values
    def sort_by_values(self, front: list, obj_list: array):
        sorted_list = []
        obj_list = deepcopy(obj_list)
        while (len(sorted_list) != len(front)):
            idx_min = argmin(obj_list)
            if idx_min in front:
                sorted_list.append(idx_min)
            obj_list[idx_min] = inf
        return sorted_list

    # Function to calculate crowding distance
    def crowding_distance(self, pop: dict, front: list):
    
        obj = [zeros(len(pop)) for i in range(self.n_objs)]
        for idx, item in enumerate(pop.values()):
            for i in range(self.n_objs):
                obj[i][idx] = float(item[self.ID_FIT][i])
        distance = [float(0.0) for _ in range(0, len(front))]
        sorted_list = []
        for i in range(self.n_objs):
            sorted_list_tmp = self.sort_by_values(front, obj[i])
            sorted_list.append(sorted_list_tmp)
        
        distance[0] = inf
        distance[len(front) - 1] = inf
        for i in range(self.n_objs):
            max_value, min_value = max(obj[i]), min(obj[i])
            diff_value = float(max_value) - float(min_value) + self.EPSILON
            for k in range(1, len(front) - 1):
                value_1 = float(obj[i][int(sorted_list[i][k + 1])])
                value_2 = float(obj[i][sorted_list[i][k - 1]])
                distance[k] += (value_1 - value_2) / diff_value
        return distance
    
    def dominate(self, id1, id2, obj):
        better = False
        for i in range(self.n_objs):
            if obj[i][id1] > obj[i][id2]:
                return False
            elif obj[i][id1] < obj[i][id2]:
                better = True
        return better

    # Function to carry out NSGA-II's fast non dominated sort
    def fast_non_dominated_sort(self, pop: dict):
        objs = [[] for _ in range(0, self.n_objs)]
        for idx, item in pop.items():
            for i in range(self.n_objs):
                objs[i].append(item[self.ID_FIT][i])
        size = len(objs[0])
        front = []
        num_assigned_individuals = 0
        indv_ranks = [0 for _ in range(0, size)]
        rank = 1
        
        while num_assigned_individuals < size:
            cur_front = []
            for i in range(size):
                if indv_ranks[i] > 0:
                    continue
                be_dominated = False
                
                j = 0
                while j < len(cur_front):
                    idx_1 = cur_front[j]
                    idx_2 = i
                    if self.dominate(idx_1, idx_2, objs):
                        be_dominated = True
                        break
                    elif self.dominate(idx_2, idx_1, objs):
                        cur_front[j] = cur_front[-1]
                        cur_front.pop()
                        j -= 1
                    j += 1
                        
                if not be_dominated:
                    cur_front.append(i)
                    
            for i in range(len(cur_front)):
                indv_ranks[ cur_front[i] ] = rank
            front.append(cur_front)
            num_assigned_individuals += len(cur_front)
            rank += 1
        return front, rank

    ## Functions for NSGA-III

    def compute_ideal_points(self, pop, fronts):
        ideal_point = [0] * self.n_objs
        conv_pop = deepcopy(pop)
        for i in range(self.n_objs):
            fit_list = [ conv_pop[list(conv_pop.keys())[stt]][self.ID_FIT][i] for stt in fronts[0]]
            ideal_point[i] = np_min(fit_list)

            for front in fronts:
                for stt in front:
                    conv_pop[list(conv_pop.keys())[stt]][self.ID_FIT][i] -= ideal_point[i]
        return ideal_point, conv_pop

    def ASF(self, objs, weight):    # Achievement Scalarization Function
        max_ratio = -inf
        for i in range(self.n_objs):
            max_ratio = max(max_ratio, objs[i] * 1.0 / weight[i])
        return max_ratio

    def find_extreme_points(self, conv_pop, fronts):
        extreme_points = []
        for f in range(self.n_objs):
            max_fit = inf
            indv = 0
            for stt_sol in range(len(conv_pop)):
                fit = conv_pop[list(conv_pop.keys())[stt_sol]][self.ID_FIT][f]
                if fit > max_fit:
                    max_fit = fit
                    indv = stt_sol
            extreme_points.append(indv)
        return extreme_points           # n_points ~ n_objs

    def get_hyperplane(self, pop, extreme_points):
        intercepts = [0.0] * self.n_objs
        ## Check duplicate points
        if len(set(extreme_points)) < len(extreme_points):
            for i in range(self.n_objs):
                idx = list(pop.keys())[extreme_points[i]]
                intercepts[i] = pop[idx][self.ID_FIT][i]
        else:
            vector_1 = ones((self.n_objs, 1))
            A = []
            for p in range(self.n_objs):
                idx = list(pop.keys())[extreme_points[p]]
                A.append(pop[idx][self.ID_FIT])
            # A = array([[-1, 1, 2], [2, 0, -3], [5, 1, -2]])        # 2D-matrix
            A = hstack((A, vector_1))
            N = len(A)
            # Khử Gauss tìm một mặt siêu phẳng
            for i in range(N - 1):
                for j in range(i + 1, N):
                    ratio = A[j][i] / A[i][i]
                    for term in range(len(A[i])):
                        A[j][term] -= A[i][term] * ratio

            x = [0.0] * N
            for i in range(N - 1, -1, -1):
                for j in range(i+1, N):
                    A[i][N] -= A[i][j] * x[j]
                x[i] = A[i][N] / A[i][i]

            for f in range(self.n_objs):
                intercepts[f] = 1.0 / x[f]
        return intercepts

    def normalize_objectives(self, pop, fronts, intercepts, ideal_points):
        conv_pop = deepcopy(pop)
        for front in fronts:
            for stt in front:
                idx = list(pop.keys())[stt]
                for f in range(self.n_objs):
                    temp = intercepts[f] - ideal_points[f]
                    if temp != 0:
                        conv_pop[idx][self.ID_FIT][f] /= temp
                    else:
                        conv_pop[idx][self.ID_FIT][f] /= self.WEIGHT        # EPSILON is too small here.
        return conv_pop

    def generate_recursive(self, rps, rpoints, left, total, element):
        if element == self.n_objs - 1:
            rpoints[element] = left * 1.0 / total
            rps.append(deepcopy(rpoints))
        else:
            for i in range(left + 1):
                rpoints[element] = i * 1.0 / total
                self.generate_recursive(rps, rpoints, left - i, total, element + 1)

    def generate_reference_points(self, num_divs):
        rpoints = [0] * self.n_objs
        rps = []
        self.generate_recursive(rps, rpoints, num_divs, num_divs, 0)
        return rps
    
    # The distance from a point to directed vector
    def perpendicular_distance(self, direction, point):
        k = dot(direction, point) / sum(power(direction, 2))
        d = sum(power(subtract(multiply(direction, [k] * len(direction)), point) , 2))
        return sqrt(d)

    def associate(self, reference_points, conv_pop, fronts, last):
        rps_pos = [[] for _ in range(0, len(reference_points))]
        num_mem = [0] * len(reference_points)
        for i in range(len(fronts)):
            front = fronts[i]
            for stt in front:
                min_rp = len(reference_points)
                min_dist = inf
                idx = list(conv_pop.keys())[stt]
                for r in range(len(reference_points)):
                    d = self.perpendicular_distance(reference_points[r], conv_pop[idx][self.ID_FIT])
                    if d < min_dist:
                        min_dist = d
                        min_rp = r
                if i < last:
                    num_mem[min_rp] += 1
                else:
                    rps_pos[min_rp].append([stt, min_dist])
        return num_mem, rps_pos

    def find_niche_reference_point(self, num_mem, rps_pos):
        # find the minimal cluster size
        cluster_members = array(num_mem)[:len(rps_pos)]
        min_size = np_min(cluster_members)
        min_rps = where(cluster_members == min_size)[0]
        if len(min_rps) == 0:
            return 0
        return min_rps[randint(0, len(min_rps) - 1)]

    def select_cluster_member(self, reference_points:list, n_mems:int, rank:list):
        chosen = -1
        if len(reference_points) > 0:
            min_rank = np_min([rank[r] for r in reference_points])
            for rp in reference_points:
                if rank[rp] == min_rank:
                    chosen = rp
        return chosen

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        pass

    def train(self):
        if Config.METRICS == "pareto" and self.__class__.__name__ not in Config.MULTI_OBJECTIVE_SUPPORTERS:
            print(f'Method: {self.__class__.__name__} doesn"t support pareto-front fitness function type')
            exit()
        print(f'Start training by: {self.__class__.__name__} algorithm, fitness type: {Config.METRICS}')

        pop_temp = [self.create_solution() for _ in range(self.pop_size)]
        self.n_objs = len(pop_temp[0][self.ID_FIT])
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
                fronts, rank = self.fast_non_dominated_sort(pop)
                current_best = []
                for it in fronts[0]:
                    current_best.append(list(pop.values())[it][self.ID_FIT])
                g_best_dict[epoch] = array(current_best)
                time_epoch_end = time() - time_epoch_start
                print(f'Front size: {len(fronts[0])}, including {list(pop.values())[fronts[0][0]][self.ID_FIT]}, '
                      f'Epoch {epoch + 1} with time: {time_epoch_end:.2f} seconds')
                if Config.TIME_BOUND_KEY:
                    if time() - time_bound_start >= Config.TIME_BOUND_VALUE:
                        print('====== Over time for training ======')
                        break
            solutions = {}
            g_best = []
            for it in fronts[0]:
                idx = list(pop.keys())[it]
                solutions[idx] = pop[idx]
                g_best.append(pop[idx][self.ID_FIT])
            return solutions, array(g_best), g_best_dict

