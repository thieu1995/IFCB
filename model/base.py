#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:10, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from abc import ABC

from utils import ToDict


class Base(ABC, ToDict):

    def __init__(self):
        self.idle_beta = 0  # power consumption - computation
        self.beta = 0  # power consumption - computation

        self.idle_alpha = 0  # power consumption - storage
        self.alpha = 0  # power consumption - storage

        self.lam_bda = 0  # latency - processing

        self.idle_pi = 0  # cost - computation
        self.pi = 0  # cost - computation

        self.idle_omega = 0  # cost - storage
        self.omega = 0  # cost - storage

    def __repr__(self):
        return str(self.to_dict())




