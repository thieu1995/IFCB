#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:32, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from config import Config
from optimizer.root import Root
from numpy import array
from numpy.random import uniform, choice, random


class GAEngine(Root):

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, time_bound=None, domain_range=None, p_c=0.9, p_m=0.05):
        super().__init__(problem, pop_size, epoch, func_eval, time_bound, domain_range)
        self.p_c = p_c
        self.p_m = p_m

    def cal_rank(self, pop):
        '''
        Calculate ranking for element in current population
        '''
        fit = []
        for i in range(len(pop)):
            fit.append(pop[i][self.ID_FIT])
        arg_rank = array(fit).argsort()
        rank = [i / sum(range(1, len(pop) + 1)) for i in range(1, len(pop) + 1)]
        return rank

    def wheel_select(self, pop, prob):
        '''
        Select dad and mom from current population by rank
        '''
        r = random()
        sum = prob[0]
        for i in range(1, len(pop) + 1):
            if sum > r:
                return i - 1
            else:
                sum += prob[i]
        return sum

    def cross_over(self, dad_element, mom_element):
        '''
        crossover dad and mom choose from current population
        '''
        r = random()
        child_element = []
        if r < self.p_c:
            is_not_valid = True
            while is_not_valid:
                for i in range(len(dad_element[0])):
                    child_element.append((dad_element[0][i] + mom_element[0][i]) / 2)
                child_schedule = matrix_to_schedule(child_element[0], child_element[1], self.fog_cloud_paths)
                if child_schedule.is_valid():
                    is_not_valid = False
            return [child_element, self.compute_fitness(child_schedule)]
        if dad_element[1] < mom_element[1]:
            if Config.METRICS == 'trade-off':
                return mom_element
            else:
                return dad_element
        else:
            if Config.METRICS == 'trade-off':
                return dad_element
            else:
                return mom_element

    def select(self, pop):
        '''
        Select from current population and create new population
        '''
        new_pop = []
        sum_fit = 0
        for i in range(len(pop)):
            sum_fit += pop[0][1]
        while len(new_pop) < self.pop_size:
            rank = self.cal_rank(pop)
            dad_index = self.wheel_select(pop, rank)
            mom_index = self.wheel_select(pop, rank)
            while dad_index == mom_index:
                mom_index = self.wheel_select(pop, rank)
            dad = pop[dad_index]
            mom = pop[mom_index]
            new_sol1 = self.cross_over(dad, mom)
            new_pop.append(new_sol1)
        return new_pop

    def mutate(self, pop):
        '''
        Mutate new population
        '''
        for i in range(len(pop)):
            is_not_valid = True
            while is_not_valid:
                for j in range(len(pop[i][0])):
                    if random() < self.p_m:
                        num_value_change = choice(range(pop[i][0][j].shape[0] * pop[i][0][j].shape[1]))
                        for k in range(num_value_change):
                            task_idx = choice(range(pop[i][0][j].shape[0]))
                            device_idx = choice(range(pop[i][0][j].shape[1]))
                            pop[i][0][j][task_idx][device_idx] = uniform(-1, 1)
                schedule = matrix_to_schedule(pop[i][0][0], pop[i][0][1], self.fog_cloud_paths)
                if schedule.is_valid():
                    is_not_valid = False
            pop[i][1] = self.compute_fitness(schedule)
        return pop

