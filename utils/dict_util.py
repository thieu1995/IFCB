#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:11, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

class ToDict:

    def to_dict(self):
        data = {}
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            data[key] = self._normalize_to_dict(value)
        return data

    def _normalize_to_dict(self, data):
        if isinstance(data, list):
            return [self._normalize_to_dict(d) for d in data]
        if isinstance(data, dict):
            return {k: self._normalize_to_dict(v) for k, v in data.items()}
        if isinstance(data, ToDict):
            return data.to_dict()
        if isinstance(data, (str, int, float)):
            return data
        return None

