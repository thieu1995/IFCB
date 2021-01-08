#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:13, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from model.base import Base


class Cloud(Base):

    def __init__(self, name=None, location=None):
        super().__init__(name, location)

        self.alpha_idle = 0     # power consumption - data forwarding - idle
        self.beta_idle = 0      # power consumption - computation - idle
        self.gamma_idle = 0     # power consumption - storage - idle

        self.sigma_idle = 0     # cost - data forwarding - idle
        self.pi_idle = 0        # cost - computation - idle
        self.omega_idle = 0     # cost - storage - idle

        self.linked_peers = []  # [peer_id, ....]
