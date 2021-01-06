#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:23, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from copy import deepcopy
import numpy as np


class Schedule:

    def __init__(self, n_clouds: int, n_fogs: int, fog_cloud_paths: np.ndarray, n_tasks: int):
        """Init the Schedule object
        n_fogs: The number of fog instances
        n_clouds: The number of cloud instances
        fog_cloud_paths: The array indicates whether is there a path from fog to cloud
        n_tasks: The number of tasks
        """

        self.n_tasks = n_tasks
        self.fog_cloud_paths = fog_cloud_paths  # ma trận ánh xạ các kết nối các nút fog lên các nút cloud

        self.cloud_schedule = [[] for _ in range(n_clouds)]
        self.fog_schedule = [[] for _ in range(n_fogs)]

    def __repr__(self):
        return f'Schedule {{\n' \
               f'  clouds: {self.cloud_schedule!r}\n' \
               f'  fogs: {self.fog_schedule!r}\n' \
               f'}}'

    def clone(self):
        return deepcopy(self)

    @property
    def total_time(self) -> int:
        cloud_time = max([len(s) for s in self.cloud_schedule])
        fog_time = max([len(s) for s in self.fog_schedule])
        return max(cloud_time, fog_time)

    def is_valid(self) -> bool:
        """
        Check whether this schedule is valid or not
        :return: bool
        """
        if self.n_tasks != len([tasks for fog in self.fog_schedule for tasks in fog]):
            return False

        if self.n_tasks != len([tasks for cloud in self.cloud_schedule for tasks in cloud]):
            return False

        inverted_cloud_schedule = [None for _ in range(self.n_tasks)]
        for cloud_id, cloud in enumerate(self.cloud_schedule):
            for task_id in cloud:
                inverted_cloud_schedule[task_id] = cloud_id

        inverted_fog_schedule = [None for _ in range(self.n_tasks)]
        for fog_id, fog in enumerate(self.fog_schedule):
            for task_id in fog:
                inverted_fog_schedule[task_id] = fog_id

        for cloud_id, fog_id in zip(inverted_cloud_schedule, inverted_fog_schedule):
            if not self.fog_cloud_paths[fog_id][cloud_id]:
                return False

        return True

