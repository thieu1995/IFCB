#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:14, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from utils.dict_util import ToDict
from uuid import uuid4
from config import DefaultData


class Task(ToDict):

    def __init__(self, r_p=0, r_s=0, q_p=0, q_s=0,
                 label=DefaultData.TASK_LABEL_IMPORTANT, sl_max=DefaultData.TASK_DEFAULT_SL_MAX):
        self.r_p = r_p
        self.r_s = r_s
        self.q_p = q_p
        self.q_s = q_s
        self.label = label      # 0: not saving to blockchain (not important), otherwise: save the blockchain
        self.sl_max = sl_max
        self.id = uuid4().hex

    # def __repr__(self):
    #     return str({
    #         'Rp': self.r_p,
    #         'Rs': self.r_s,
    #         'Qp': self.q_p,
    #         'Qs': self.q_s,
    #         'Label': self.label,
    #         'SL_MAX': self.sl_max
    #     })

    def __repr__(self):
        return str(self.to_dict())

    def __eq__(self, other):
        return self.id == other.id and self.sl_max == other.sl_max

    def __hash__(self):
        return hash(('id', self.id, 'sl_max', self.sl_max))
