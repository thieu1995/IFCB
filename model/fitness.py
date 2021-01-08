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


class Fitness:

    def __init__(self, problem):
        self.clouds = problem["clouds"]
        self.fogs = problem["fogs"]
        self.peers = problem["peers"]
        self.tasks = problem["task"]
        self.list_to_dict(self.clouds, self.fogs, self.peers, self.tasks)

        self._min_power = 0
        self._min_latency = 0
        self._min_cost = 0
        self.alpha_trade_off = 1 / 3
        self.beta_trade_off = 1 / 3

    def list_to_dict(self, *attributes):
        for idx, attr in enumerate(attributes):
            objs = getattr(self, attr)
            obj_list = {}
            for obj in objs:
                obj_list[obj.id] = obj
            setattr(self, attr, obj_list)

    def set_min_power(self, value: float):
        self._min_power = value

    def set_min_latency(self, value: float):
        self._min_latency = value

    def set_min_cost(self, value: float):
        self._min_cost = value

    def set_trade_off(self, trade_off_value):
        self.alpha_trade_off = trade_off_value[0]
        self.beta_trade_off = trade_off_value[1]
        assert self.alpha_trade_off >= 0
        assert self.beta_trade_off >= 0
        assert self.alpha_trade_off + self.beta_trade_off <= 1.0

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
        elif Config.METRICS == 'trade-off':
            # print('power information: ', self._min_power, power, self._min_power / power)
            # print('latency information: ', self._min_latency, latency, self._min_latency / latency)
            # print('cost information: ', self._min_cost, cost, self._min_cost / cost)
            # print('------------------------')
            return self.alpha_trade_off * (self._min_power / power) \
                   + self.beta_trade_off * (self._min_latency / latency) \
                   + (1 - self.alpha_trade_off - self.beta_trade_off) * (self._min_cost / cost)
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

