#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 03:12, 14/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from optimizer.root2 import Root2
from numpy.random import choice
from numpy import array, where
from uuid import uuid4


class BaseNSGA_II(Root2):
    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        if paras is None:
            paras = {"p_c": 0.9, "p_m": 0.1}
        self.p_c = paras["p_c"]
        self.p_m = paras["p_m"]

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        # Generating offsprings
        pop_temp = {}
        while (len(pop_temp) != 2 * self.pop_size):
            stt_dad, stt_mom = choice(list(range(0, self.pop_size)), 2, replace=False)
            idx_dad, idx_mom = list(pop.keys())[stt_dad], list(pop.keys())[stt_mom]
            temp = self.crossover(pop[idx_dad], pop[idx_mom], self.p_c)
            temp = self.mutate(temp, self.p_m)
            pop_temp[temp[self.ID_IDX]] = temp

        non_dominated_list, ranks = self.fast_non_dominated_sort(pop_temp)
        cdist_lists = []
        for i in range(0, len(non_dominated_list)):
            cdist_lists.append(self.crowding_distance(pop_temp, non_dominated_list[i]))

        pop = {}
        for i in range(0, len(non_dominated_list)):
            current_NDS_list = array(non_dominated_list[i])
            NDS_list_child = [where(current_NDS_list == current_NDS_list[j]) for j in range(0, len(current_NDS_list))]
            front22 = self.sort_by_values(NDS_list_child, cdist_lists[i])
            front = [current_NDS_list[front22[j]] for j in range(0, len(current_NDS_list))]
            front.reverse()
            for idx in front:
                idx_real = list(pop_temp.keys())[idx]
                _idx = uuid4().hex
                pop[_idx] = pop_temp[idx_real]
                if len(pop) == self.pop_size:
                    break
            if len(pop) == self.pop_size:
                break
        return pop
