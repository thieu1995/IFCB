#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:14, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from utils import ToDict


class Task(ToDict):

    def __init__(self):
        self.p_r = 0
        self.p_s = 0
        self.q_r = 0
        self.q_s = 0

    def __repr__(self):
        return str({
            'Pr': self.p_r,
            'Ps': self.p_s,
            'Qr': self.q_r,
            'Qs': self.q_s,
        })

