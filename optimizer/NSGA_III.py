#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 03:12, 14/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from copy import deepcopy
from optimizer.root2 import Root2
from numpy.random import choice
from uuid import uuid4


class BaseNSGA_III(Root2):
    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub)
        if paras is None:
            paras = {"p_c": 0.9, "p_m": 0.1, "cof_divs": 12, "old_pop_rate": 0.7}
        self.p_c = paras["p_c"]
        self.p_m = paras["p_m"]
        self.cof_divs = paras["cof_divs"]
        self.old_pop_rate = paras["old_pop_rate"]

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        
        fronts, rank = self.fast_non_dominated_sort(pop)
        pop_temp = {}
        
        for front in fronts:
            stop = False
            for id in front:
                key = list(pop.keys())[id]
                _idx = uuid4().hex
                pop_temp[_idx] = pop[key]
                if len(pop_temp) == self.old_pop_rate * self.pop_size:
                    stop = True
                    break
            if stop:
                break
            
        # Generating offsprings
        while (len(pop_temp) < 2 * self.pop_size):
            stt_dad, stt_mom = choice(list(range(0, self.pop_size)), 2, replace=False)
            idx_dad, idx_mom = list(pop.keys())[stt_dad], list(pop.keys())[stt_mom]
            child = self.crossover(pop[idx_dad], pop[idx_mom], self.p_c)
            child = self.mutate(child, self.p_m)
            pop_temp[child[self.ID_IDX]] = child
        pop = deepcopy(pop_temp)

        fronts, rank = self.fast_non_dominated_sort(pop)
        last = 0
        next_size = 0
        new_pop = {}

        while next_size < self.pop_size:
            next_size += len(fronts[last])
            last += 1
            if last == len(fronts):
                break

        for i in range(len(fronts) - 1):
            for idx in fronts[i]:
                key = list(pop.keys())[idx]
                _idx = uuid4().hex
                new_pop[_idx] = deepcopy(pop[key])
                new_pop[_idx][self.ID_IDX] = _idx
        
        if len(new_pop) == self.pop_size:
            return new_pop
        ## ideal_point: la min(fit_list)
        ## conv_pop: change the fitness cua solution in fronts
        ideal_point, conv_pop = self.compute_ideal_points(pop, fronts)

        ### N_global best solution for each objective
        extreme_points = self.find_extreme_points(conv_pop, fronts)

        ### The fitness of n-global-best (if duplicate --> will use Gauss)
        intercepts = self.get_hyperplane(conv_pop, extreme_points)

        ## Normalize fitness of population
        conv_pop = self.normalize_objectives(conv_pop, fronts, intercepts, ideal_point)

        ### Generate ref points
        reference_points = self.generate_reference_points(self.cof_divs)

        # Divide the indvs to diff cluster by ref vector
        num_mem, rps_pos = self.associate(reference_points, conv_pop, fronts, last)
        
        while len(new_pop) < self.pop_size:
            min_rp = self.find_niche_reference_point(num_mem, rps_pos)
            chosen = self.select_cluster_member(rps_pos[min_rp], num_mem[min_rp], rank)
            if chosen < 0:
                rps_pos.pop(min_rp)
                num_mem.pop(min_rp)
            else:
                num_mem[min_rp] += 1
                for i in range(len(rps_pos[min_rp])):
                    if rps_pos[min_rp][i][0] == chosen:
                        rps_pos[min_rp].pop(i)
                        num_mem[min_rp].pop(i)
                        break
                idx = list(pop.keys())[chosen]
                _idx = uuid4().hex
                new_pop[_idx] = deepcopy(pop[idx])
                new_pop[_idx][self.ID_IDX] = _idx

        return deepcopy(new_pop)
