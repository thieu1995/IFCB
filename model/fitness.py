#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:43, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from config import *
from .schedule import Schedule
from model.formulas import power, latency, cost
from numpy import array
from numpy.linalg import norm


class Fitness:

    def __init__(self, problem):
        self.clouds = problem["clouds"]
        self.fogs = problem["fogs"]
        self.peers = problem["peers"]
        self.tasks = problem["tasks"]
        self.list_to_dict({'clouds': self.clouds, 'fogs': self.fogs, 'peers': self.peers, 'tasks': self.tasks})

        self._min_power = 0
        self._min_latency = 0
        self._min_cost = 0

    def list_to_dict(self, *attributes):
        for idx, attrs in enumerate(attributes):
            for key, values in attrs.items():
                obj_list = {}
                for obj in values:
                    obj_list[obj.id] = obj
                setattr(self, key, obj_list)

    def set_min_power(self, value: float):
        self._min_power = value

    def set_min_latency(self, value: float):
        self._min_latency = value

    def set_min_cost(self, value: float):
        self._min_cost = value

    def fitness(self, solution: Schedule) -> float:
        power = self.calc_power_consumption(solution)
        latency = self.calc_latency(solution)
        cost = self.calc_cost(solution)

        # assert self._min_power <= power
        # assert self._min_latency <= latency
        # assert self._min_cost <= cost

        if Config.METRICS == 'power':
            return power
        elif Config.METRICS == 'latency':
            return latency
        elif Config.METRICS == 'cost':
            return cost
        elif Config.METRICS == "weighting":
            w = array(Config.OBJ_WEIGHTING_METRICS)
            v = array([power, latency, cost])
            return sum(w * v)
        elif Config.METRICS == "distancing":
            o = array(Config.OBJ_DISTANCING_METRICS)
            v = array([power, latency, cost])
            return norm(o-v)
        elif Config.METRICS == 'min-max':
            o = array(Config.OBJ_MINMAX_METRICS)
            v = array([power, latency, cost])
            return max((v-o)/o)         # Need to minimize the relative deviation of single objective functions
        elif Config.METRICS == "weighting-min":   # The paper of Thang and Khiem
            w = array(Config.OBJ_WEIGHTING_MIN_METRICS_1)
            o = array(Config.OBJ_WEIGHTING_MIN_METRICS_2)
            v = array([power, latency, cost])
            return sum((w * o) / v)
        else:
            print(f'[ERROR] Metrics {Config.METRICS} is not supported in class FitnessManager')

    def calc_power_consumption(self, schedule: Schedule) -> float:
        po = power.data_forwarding_power(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        po += power.computation_power(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        po += power.storage_power(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        return po / 3600

    def calc_latency(self, schedule: Schedule) -> float:
        la = latency.processing_latency(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        la += latency.processing_latency(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        return la

    def calc_cost(self, schedule: Schedule) -> float:
        co = cost.data_forwarding_cost(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        co += cost.computation_cost(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        co += cost.storage_cost(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        return co

