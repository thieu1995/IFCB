#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:10, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from abc import ABC
from utils.dict_util import ToDict
from numpy import array, sqrt, sum
from json import dumps
from uuid import uuid4


class Base(ABC, ToDict):

    def __init__(self, name:str, location:array) -> None:
        self.name = name
        self.location = location
        self.id = uuid4().hex

        self.alpha = 0      # power consumption - data forwarding
        self.beta = 0       # power consumption - computation
        self.gamma = 0      # power consumption - storage

        self.eta = 0        # latency - transmission
        self.lamda = 0      # latency - processing

        self.sigma = 0      # cost - data forwarding
        self.pi = 0         # cost - computation
        self.omega = 0      # cost - storage

    def __repr__(self):
        return str(self.to_dict())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id and self.name == other.name and self.location == other.location

    def __hash__(self):
        return hash((self.id, self.name, dumps(self.location)))

    def dist(self, other) -> float:
        return sqrt(sum((self.location.get(d, 0) - other.location.get(d, 0)) ** 2 for d in set(self.location) | set(other.location)))
