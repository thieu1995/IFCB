#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:13, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from .base import Base


class Fog(Base):

    def __init__(self):
        super().__init__()

        self.idle_eg_gamma = 0  # power consumption - data forwarding
        self.eg_gamma = 0  # power consumption - data forwarding

        self.idle_fi_gamma = 0  # power consumption - data forwarding
        self.fi_gamma = 0  # power consumption - data forwarding

        self.idle_cl_gamma = []  # power consumption - data forwarding
        self.cl_gamma = []  # power consumption - data forwarding

        self.ef_delta = 0  # latency - transmission
        self.fg_delta = []  # latency - transmission

        self.idle_eg_sigma = 0  # cost - data forwarding
        self.eg_sigma = 0  # cost - data forwarding

        self.idle_fi_sigma = 0  # cost - data forwarding
        self.fi_sigma = 0  # cost - data forwarding

        self.idle_cl_sigma = []  # cost - data forwarding
        self.cl_sigma = []  # cost - data forwarding

        self.tau = 10_000  # time-to-live
