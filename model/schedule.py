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
from numpy import ndarray
from typing import List

from model import formulas
from model.cloud import Cloud
from model.fog import Fog
from model.blockchain.node import Node
from model.task import Task


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

        self.schedule_clouds_tasks = [
            # {
            #     "cloud_id": 1,
            #     "list_task_id": [1, 2, 3, 4],
            # },
            # {
            #     "cloud_id": 2,
            #     "list_task_id": [5, 7],
            # }
        ]
        self.schedule_fogs_tasks = [
            # {
            #     "fog_id": 1,
            #     "list_task_id": [1, 2, 3, 4],
            # },
            # {
            #     "fog_id": 2,
            #     "list_task_id": [5, 7],
            # }
        ]
        self.schedule_peers_tasks = [
            # {
            #     "peer_id": 1,
            #     "list_task_id": [1, 2, 3, 4],
            # },
            # {
            #     "peer_id": 2,
            #     "list_task_id": [5, 7],
            # }
        ]

    def get_list_task_handlers(self):
        task_handlers = {
            # "task_id": ["fog_id", "cloud_id"],
            # "2": [3, 4],
        }
        for obj in self.schedule_fogs_tasks:
            for task_id in obj["list_task_id"]:
                if task_id not in task_handlers.keys():
                    task_handlers[task_id] = [obj["fog_id"]]
                else:
                    task_handlers[task_id].append(obj["fog_id"])
        for obj in self.schedule_clouds_tasks:
            for task_id in obj["list_task_id"]:
                if task_id not in task_handlers.keys():
                    task_handlers[task_id] = [obj["cloud_id"]]
                else:
                    task_handlers[task_id].append(obj["cloud_id"])
        return task_handlers

    def is_valid(self) -> bool:
        """
        Check whether this schedule is valid or not
        :return: bool
        """

        ## 1. Total tasks in fogs = total tasks in clouds = total tasks
        tasks_temp = [obj["list_task_id"] for obj in self.schedule_clouds_tasks]
        tasks = list(chain(*tasks_temp))
        if len(set(tasks)) != self.n_tasks:
            return False

        tasks_temp = [obj["list_task_id"] for obj in self.schedule_fogs_tasks]
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
        cloud_time = max([len(item["list_task_id"]) for item in self.schedule_clouds_tasks])
        fog_time = max([len(item["list_task_id"]) for item in self.schedule_fogs_tasks])
        return max(cloud_time, fog_time)