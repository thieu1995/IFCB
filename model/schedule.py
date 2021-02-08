#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:23, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from itertools import chain
from copy import deepcopy


class Schedule:

    def __init__(self, problem):
        """Init the Schedule object
        n_fogs: The number of fog instances
        n_clouds: The number of cloud instances
        n_peers: The number of blockchain nodes
        n_tasks: The number of tasks
        """
        self.clouds = problem["clouds"]
        self.fogs = problem["fogs"]
        self.peers = problem["peers"]
        self.tasks = problem["tasks"]

        self.n_clouds = len(self.clouds)
        self.n_fogs = len(self.fogs)
        self.n_peers = len(self.peers)
        self.n_tasks = len(self.tasks)

        self.schedule_clouds_tasks = {} # key: cloud_id, val: list_task_id []
        self.schedule_fogs_tasks = {}
        self.schedule_peers_tasks = {}

    def get_list_task_handlers(self):
        task_handlers = {
            # "task_id": ["fog_id", "cloud_id"],
            # "2": [3, 4],
        }
        for fog_id, list_task_id in self.schedule_fogs_tasks.items():
            for task_id in list_task_id:
                if task_id not in task_handlers.keys():
                    task_handlers[task_id] = [fog_id]
                else:
                    task_handlers[task_id].append(fog_id)
        for cloud_id, list_task_id in self.schedule_clouds_tasks.items():
            for task_id in list_task_id:
                if task_id not in task_handlers.keys():
                    task_handlers[task_id] = [cloud_id]
                else:
                    task_handlers[task_id].append(cloud_id)
        return task_handlers

    def get_list_task_handlers_with_order(self):
        task_handlers = {
            # "task_id": {
            #       "fog_id": [taskId, taskID,...],
            #       "cloud_id": [taskId, taskId, ...], those tasks which will be handled before this task_id
            # },
        }
        for fog_id, list_task_id in self.schedule_fogs_tasks.items():
            for idx, task_id in enumerate(list_task_id):
                if task_id not in task_handlers.keys():
                    list_task_before = []
                    for idx_before in range(0, idx):
                        list_task_before.append(list_task_id[idx_before])
                    task_handlers[task_id] = {
                        fog_id: list_task_before
                    }
                else:
                    continue
        for cloud_id, list_task_id in self.schedule_clouds_tasks.items():
            for idx, task_id in enumerate(list_task_id):
                list_task_before = []
                for idx_before in range(0, idx):
                    list_task_before.append(list_task_id[idx_before])
                task_handlers[task_id][cloud_id] = list_task_before
        return task_handlers

    def is_valid(self) -> bool:
        """
        Check whether this schedule is valid or not
        :return: bool
        """

        ## 1. Total tasks in fogs = total tasks in clouds = total tasks
        tasks_temp = [task_id for task_id in self.schedule_clouds_tasks.values()]
        tasks = list(chain(*tasks_temp))        ## Kinda same as set(list) to remove duplicate element
        if len(set(tasks)) != self.n_tasks:
            return False

        tasks_temp = [task_id for task_id in self.schedule_fogs_tasks.values()]
        tasks = list(chain(*tasks_temp))
        if len(set(tasks)) != self.n_tasks:
            return False

        ## 2. The task handle by fog-cloud, but there is no linked between fog-cloud --> in-valid schedule
        task_handlers = self.get_list_task_handlers()
        for th_key, th_val in task_handlers.items():
            if len(th_val) != 2:
                return False
            fog_id, cloud_id = th_val
            fog = next((x for x in self.fogs if x.id == fog_id), None)
            if fog is None or cloud_id not in fog.linked_clouds:
                return False

        ## 3. The task saving in blockchain has label "0" -- not important --> in-valid schedule
        return True

    def __repr__(self):
        return f'Schedule {{\n' \
               f'  clouds: {self.schedule_clouds_tasks!r}\n' \
               f'  fogs: {self.schedule_fogs_tasks!r}\n' \
               f'}}'

    def clone(self):
        return deepcopy(self)

    @property
    def total_time(self) -> int:
        cloud_time = max([len(val) for val in self.schedule_clouds_tasks.values()])
        fog_time = max([len(val) for val in self.schedule_fogs_tasks.values()])
        return max(cloud_time, fog_time)