#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 03:12, 14/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from copy import deepcopy
from config import Config
from optimizer.root3 import Root3
from numpy.random import uniform, random, choice
from numpy import array, where
from utils.schedule_util import matrix_to_schedule
from uuid import uuid4


class BaseNSGA_III(Root3):

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub)
        if paras is None:
            paras = {"p_c": 0.9, "p_m": 0.05}
        self.p_c = paras["p_c"]
        self.p_m = paras["p_m"]
        self.num_objectives = 3

    def crossover(self, dad, mom):
        r = random()
        if r < self.p_c:
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

    def mutate(self, child):
        while True:
            child_pos = deepcopy(child[self.ID_POS])
            rd_matrix = uniform(self.lb, self.ub, self.problem["shape"])
            child_pos[rd_matrix < self.p_m] = 0
            rd_matrix_new = uniform(self.lb, self.ub, self.problem["shape"])
            rd_matrix_new[rd_matrix >= self.p_m] = 0
            child_pos = child_pos + rd_matrix_new
            schedule = matrix_to_schedule(self.problem, child_pos)
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                idx = uuid4().hex
                break
        return [idx, child_pos, fitness]

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        
        pop_temp = deepcopy(pop)

        # Generating offsprings
        while (len(pop_temp) != 2 * self.pop_size):
            stt_dad, stt_mom = choice(list(range(0, self.pop_size)), 2, replace=False)
            idx_dad, idx_mom = list(pop.keys())[stt_dad], list(pop.keys())[stt_mom]
            temp = self.crossover(pop[idx_dad], pop[idx_mom])
            temp = self.mutate(temp)
            pop_temp[temp[self.ID_IDX]] = temp
            
        pop = deepcopy(pop_temp)
            
        fronts = self.fast_non_dominated_sort(pop)
        last = 0
        next_size = 0;
        new_pop = {}
        
        print(fronts)
        
        while (next_size < self.pop_size):
            next_size += len(fronts[last])
            last += 1
            if (last == len(fronts)):
                break
        while(len(fronts) > last):
            fronts.pop() # remove useless individual

        for i in range(len(fronts) - 1):
            for idx in fronts[i]:
                key = list(pop.keys())[idx]
                _idx = uuid4().hex
                new_pop[_idx] = deepcopy(pop[key])
        
        if len(new_pop) == self.pop_size:
            return new_pop
        ideal_point, conv_pop = self.compute_ideal_points(pop, fronts, self.num_objectives)
        
        extreme_points = self.find_extreme_points(pop, fronts)
        
        intercepts = self.get_hyperplane(pop, extreme_points)
        
        conv_pop = self.normalize_objectives(pop, fronts, intercepts, ideal_point)
        
        rps = self.generate_reference_points(12)

        num_mem, rps_pos = self.associate(rps, conv_pop, fronts)
        while len(new_pop) < self.pop_size:
            min_rp = self.FindNicheReferencePoint(num_mem, rps_pos)
            chosen = self.SelectClusterMember(rps_pos[min_rp], num_mem[min_rp])
            if chosen < 0:
                rps_pos.pop(min_rp)
            else:
                num_mem[min_rp] += 1
                for i in range(len(rps_pos[min_rp])):
                    if(rps_pos[min_rp][i][0] == chosen):
                        rps_pos[min_rp].pop(i)
                        break
                key = list(pop.keys())[chosen]
                _idx = uuid4().hex
                new_pop[_idx] = deepcopy(pop[key])
        
        pop = new_pop
        return pop
