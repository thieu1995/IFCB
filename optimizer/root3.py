#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:35, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from optimizer.root import Root
from numpy import array, inf
from numpy.random import uniform
from time import time
from copy import deepcopy
from math import sqrt
from random import randint
from utils.schedule_util import matrix_to_schedule
from uuid import uuid4
from config import Config


class Root3(Root):
    ID_IDX = 0
    ID_POS = 1
    ID_FIT = 2

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub)
        self.num_obj = 3

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
                    if self.dominate(idx_1, idx_2, obj):
                        be_dominated = True
                        break
                    elif self.dominate(idx_2, idx_1, obj):
                        cur_front[j] = cur_front[-1]
                        cur_front.pop()
                        j -= 1
                    j += 1

                if (not be_dominated):
                    cur_front.append(i)

            for i in range(len(cur_front)):
                indv_ranks[cur_front[i]] = rank
            front.append(cur_front)
            num_assigned_individuals += len(cur_front)
            rank += 1
        return front

    def create_solution(self):
        while True:
            matrix = uniform(self.lb, self.ub, self.problem["shape"])
            schedule = matrix_to_schedule(self.problem, matrix)
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                idx = uuid4().hex
                break
        return [idx, matrix, fitness]

    def compute_ideal_points(self, pop, fronts, num_obj):
        ideal_point = [0] * num_obj
        conv_pop = deepcopy(pop)
        for i in range(num_obj):
            
            minf = 1e10
            
            for j in range(len(fronts[0])):
                idx = list(conv_pop.keys())[fronts[0][j]]
                value = conv_pop[idx][self.ID_FIT][i]
                minf = min(minf, value)
            
            ideal_point[i] = minf
            
            for j in range(len(fronts)):
                for k in range(len(fronts[j])):
                    idx = list(conv_pop.keys())[fronts[j][k]]
                    conv_pop[idx][self.ID_FIT][i] -= minf

        print(ideal_point)
        return ideal_point, conv_pop

    def ASF(self, objs, weight):
        max_ratio = -inf
        for i in range(self.num_obj):
            w = 0.0
            if weight[i] != 0:
                w = weight[i]
            else:
                w = 0.00001
            max_ratio = max(max_ratio, objs[i] * 1.0 / w)
        return max_ratio

    def find_extreme_points(self, pop, fronts):
        
        extreme_points = []
        
        for f in range(self.num_obj):    
            w = [0.000001] * self.num_obj
            w[f] = 1.0
            min_ASF = inf
            min_indv = len(fronts[0])
            
            for j in range(len(fronts[0])):
                idx = list(pop.keys())[fronts[0][j]]
                asf = self.ASF(pop[idx][self.ID_FIT], w)
                if asf < min_ASF:
                    min_ASF = asf
                    min_indv = fronts[0][j]
                    
            extreme_points.append(min_indv)
            
        return extreme_points        
        
    def get_hyperplane(self, pop, extreme_points):
        
        intercepts = [0.0] * (self.num_obj)
        duplicate = False
        for i in range(len(extreme_points)):
            for j in range(i, len(extreme_points)):
                if (duplicate):
                    break
                duplicate = (extreme_points[i] == extreme_points[j])
        ## Loi vong lap for ben tren, luon tra gia tri True


        if duplicate:
            for i in range(len(intercepts)):
                idx = list(pop.keys())[extreme_points[i]]
                intercepts[i] = pop[idx][self.ID_FIT][i]
        else:
            b = [0.0] * self.num_obj
            A = [[]]
            for p in range(len(extreme_points)):
                idx = list(pop.keys())[extreme_points[p]]
                A.append(pop[idx][self.ID_FIT])
              
            # Khử Gauss tìm một mặt siêu phẳng
            N = len(A)
            for p in range(N):
                A[p].append(b[p])
            for i in range(N):
                for j in range(i + 1, N):
                    ratio = A[j][i] / A[i][i]
                    for term in range(len(A[i])):
                        A[j][term] -= A[i][term] * ratio
                        
            x = [0.0] * N
            for i in range(N - 1, 0, -1):
                for j in range(N):
                    A[i][len] -= A[i][j] * x[j]
                x[i] = A[i][N] / A[i][i]
                
            for f in range(self.num_obj):
                intercepts[f] = 1.0 / x[f]
                
        return intercepts        
        
    def normalize_objectives(self, pop, fronts, intercepts, ideal_points):
        conv_pop = deepcopy(pop)
        for t in range(len(fronts)):
            for i in range(len(fronts[t])):
                ind = list(pop.keys())[fronts[t][i]]
                for f in range(self.num_obj):
                    if intercepts[f] - ideal_points[f] != 0:
                        conv_pop[ind][self.ID_FIT][f] /= intercepts[f] - ideal_points[f]
                    else:
                        conv_pop[ind][self.ID_FIT][f] /= 0.00001
        return conv_pop
    
    def generate_recursive(self, rps, rpoints, left, total, element):
        if (element == self.num_obj - 1):
            rpoints[element] = left * 1.0 / total
            rps.append(deepcopy(rpoints))
        else:
            for i in range(left + 1):
                rpoints[element] = i * 1.0 / total
                self.generate_recursive(rps, rpoints, left - i, total, element + 1)
                
    def generate_reference_points(self, num_divs):
        rpoints = [0] * self.num_obj
        rps = []
        self.generate_recursive(rps, rpoints, num_divs, num_divs, 0)
        return rps
    
    def PerpendicularDistance(self, direction, point):
        
        numerator = 0
        denominator = 0
        for i in range(len(direction)):
            numerator += direction[i] * point[i]
            denominator += direction[i] ** 2
        
        k = numerator * 1.0 / denominator
        d = 0
        for i in range(len(direction)):
            d += (k * direction[i] - point[i]) ** 2
        
        return sqrt(d)
    
    def associate(self, rps, conv_pop, fronts):
        rps_pos = [[] for _ in range(0, len(rps))] 
        num_mem = [0] * len(rps)
        
        for t in range(len(fronts)):
            for i in range(len(fronts[t])):
                min_rp = len(rps)
                min_dist = inf
                idx = list(conv_pop.keys())[fronts[t][i]]
                for r in range(len(rps)):
                    d = self.PerpendicularDistance(rps[r], conv_pop[idx][self.ID_FIT])  
                    if d < min_dist:
                        min_dist = d
                        min_rp = r
                num_mem[min_rp] += 1
                rps_pos[min_rp].append([fronts[t][i], min_dist])
        return num_mem, rps_pos
    
    def FindNicheReferencePoint(self, num_mem, rps_pos):
        # find the minimal cluster size
        min_size = inf
        for r in range(len(rps_pos)):
            min_size = min(min_size, num_mem[r])
        
        min_rps = []
        for r in range(len(rps_pos)):
            if num_mem[r] == min_size:
                min_rps.append(r)
        if len(min_rps) == 0:
            return -1
        
        return min_rps[randint(0, len(min_rps) - 1)]
    
    def FindClosestMember(self, potentials):
        min_dist = inf
        min_indv = -1
        for i in range(len(potentials)):
            if potentials[i][1] < min_dist:
                min_dist = potentials[i][1]
                min_indv = potentials[i][0]
            
        return min_indv
    
    def RandomMember(self, potentials):
        if(len(potentials) == 0):
            return -1
        return potentials[randint(0, len(potentials) - 1)][0]
    
    def SelectClusterMember(self, rp, n_mems):
        chosen = -1
        if (len(rp) > 0):
            if n_mems == 0: # currently has no member:
                chosen =  self.FindClosestMember(rp)
            else:
                chosen =  self.RandomMember(rp)
        return chosen
    
    def dominate(self, id1, id2, obj):
        better = False
        for i in range(self.num_obj):
            if obj[i][id1] > obj[i][id2]:
                return False
            elif obj[i][id1] < obj[i][id2]:
                better = True
        return better

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