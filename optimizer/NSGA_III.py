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
            paras = {"p_c": 0.9, "p_m": 0.05}
        self.p_c = paras["p_c"]
        self.p_m = paras["p_m"]

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        pop_temp = deepcopy(pop)
        # Generating offsprings
        while (len(pop_temp) != 2 * self.pop_size):
            stt_dad, stt_mom = choice(list(range(0, self.pop_size)), 2, replace=False)
            idx_dad, idx_mom = list(pop.keys())[stt_dad], list(pop.keys())[stt_mom]
            child = self.crossover(pop[idx_dad], pop[idx_mom], self.p_c)
            child = self.mutate(child, self.p_m)
            pop_temp[child[self.ID_IDX]] = child
        pop = deepcopy(pop_temp)

        fronts = self.fast_non_dominated_sort(pop)
        last = 0
        next_size = 0
        new_pop = {}

        while next_size < self.pop_size:
            next_size += len(fronts[last])
            last += 1
            if last == len(fronts):
                break
        while len(fronts) > last:
            fronts.pop()        # remove useless individual

        for i in range(len(fronts) - 1):
            for idx in fronts[i]:
                key = list(pop.keys())[idx]
                _idx = uuid4().hex
                new_pop[_idx] = deepcopy(pop[key])
        
        if len(new_pop) == self.pop_size:
            return new_pop
        ## ideal_point: la min(fit_list) cua front0
        ## conv_pop: change the fitness cua solution in fronts
        ideal_point, conv_pop = self.compute_ideal_points(pop, fronts)

        ### N_global best solution for each objective
        extreme_points = self.find_extreme_points(pop, fronts)

        ### The fitness of n-global-best (if duplicate --> will use Gauss)
        intercepts = self.get_hyperplane(pop, extreme_points)

        ## Ben tren return conv_pop nhung ben duoi lai khong dung den???
        conv_pop = self.normalize_objectives(pop, fronts, intercepts, ideal_point)

        ### Kinda like creating brute-force weights for all objectives???
        reference_points = self.generate_reference_points(6)

        num_mem, rps_pos = self.associate(reference_points, conv_pop, fronts)
        while len(new_pop) < self.pop_size:
            min_rp = self.find_niche_reference_point(num_mem, rps_pos)
            chosen = self.select_cluster_member(rps_pos[min_rp], num_mem[min_rp])
            if chosen < 0:
                rps_pos.pop(min_rp)
            else:
                num_mem[min_rp] += 1
                for i in range(len(rps_pos[min_rp])):
                    if rps_pos[min_rp][i][0] == chosen:
                        rps_pos[min_rp].pop(i)
                        break
                idx = list(pop.keys())[chosen]
                new_pop[idx] = deepcopy(pop[idx])

        return deepcopy(new_pop)
