#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:43, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from typing import List
from model import Cloud, Fog, Task
from config import *
from optimizer import formulas
from .schedule import Schedule


class Fitness:

    def __init__(self, clouds: List[Cloud], fogs: List[Fog], tasks: List[Task]):
        self.clouds = clouds
        self.fogs = fogs
        self.tasks = tasks

        self._min_power = 0
        self._min_latency = 0
        self._min_cost = 0
        self.alpha_trade_off = 1 / 3
        self.beta_trade_off = 1 / 3

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

    def calc(self, schedule: Schedule) -> float:
        power = self.calc_power_consumption(schedule)
        latency = self.calc_latency(schedule)
        cost = self.calc_cost(schedule)

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
        power = formulas.data_forwarding_power(self.clouds, self.fogs, self.tasks, schedule)
        power += formulas.computation_power(self.clouds, self.fogs, self.tasks, schedule)
        power += formulas.storage_power(self.clouds, self.fogs, self.tasks, schedule)
        return power / 3600

    def calc_latency(self, schedule: Schedule) -> float:
        latency = formulas.processing_latency(self.clouds, self.fogs, self.tasks, schedule)
        latency += formulas.processing_latency(self.clouds, self.fogs, self.tasks, schedule)
        return latency

    def calc_cost(self, schedule: Schedule) -> float:
        cost = formulas.data_forwarding_cost(self.clouds, self.fogs, self.tasks, schedule)
        cost += formulas.computation_cost(self.clouds, self.fogs, self.tasks, schedule)
        cost += formulas.storage_cost(self.clouds, self.fogs, self.tasks, schedule)
        return cost

