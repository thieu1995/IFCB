#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 19:38, 07/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from model.base import Base


class Node(Base):

    def __init__(self, name=None, location=None):
        super().__init__(name, location)

        self.alpha_sm = 0       # power consumption - data forwarding - standby mode
        self.gamma_sm = 0       # power consumption - storage - standby mode

        self.sigma_sm = 0       # cost - data forwarding - standby mode
        self.omega_sm = 0       # cost - storage - standby mode

