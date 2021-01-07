#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:13, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from .base import Base
from numpy import array


class Fog(Base):

    def __init__(self, id:int, name:str, location:array):
        super().__init__(id, name, location)

        self.alpha_idle = 0     # power consumption - data forwarding - idle
        self.beta_idle = 0      # power consumption - computation - idle
        self.gamma_idle = 0     # power consumption - storage - idle

        self.sigma_idle = 0     # cost - data forwarding - idle
        self.pi_idle = 0        # cost - computation - idle
        self.omega_idle = 0     # cost - storage - idle

        self.alpha_device = 0       # power consumption - data forwarding - end device to fog
        self.alpha_device_idle = 0  # power consumption - data forwarding

        self.sigma_device = 0       # cost - data forwarding - end device to fog
        self.sigma_device_idle = 0  # cost - data forwarding - idle state

        self.tau = 10_000  # time-to-live

        self.linked_clouds = []
        self.linked_peers = []
