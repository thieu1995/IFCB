#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:23, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import random

import numpy as np

from model import Schedule
from utils import load_cloudlets, load_tasks

clouds, fogs = load_cloudlets('data/cloudlet_5_20.json')
tasks = load_tasks('data/tasks_1000.json')

fog_cloud_paths = np.array([fog.idle_cl_gamma for fog in fogs]) != float('inf')


def create_schedule():
    s = Schedule(len(clouds), len(fogs), fog_cloud_paths, len(tasks))
    for task_id in range(len(tasks)):
        fog_idx = random.randrange(len(fogs))
        s.fog_schedule[fog_idx].append(task_id)

        fog_idx = random.randrange(len(fogs))
        s.cloud_paths[task_id] = fog_idx

        fog = fogs[fog_idx]
        cloud_ids = []
        for cloud_id, p in enumerate(fog.cl_gamma):
            if p != float('inf'):
                cloud_ids.append(cloud_id)

        cloud_idx = random.choice(cloud_ids)
        s.cloud_schedule[cloud_idx].append(task_id)
    return s

# print(s)
# print(s.is_valid())

# from algorithms import formulas

# a = formulas.data_forwarding_power(clouds, fogs, tasks, schedule=s)
# print(a)


def matrix_to_schedule(cloud_matrix: np.ndarray, fog_matrix: np.ndarray, fog_cloud_paths: np.ndarray) -> Schedule:
    """
    Convert matrix data to schedule object
    :param cloud_matrix: n_task x n_cloud
    :param fog_matrix: n_task x n_fog
    :return: Schedule object
    """

    n_clouds = cloud_matrix.shape[1]
    n_fogs = fog_matrix.shape[1]
    n_tasks = cloud_matrix.shape[0]

    schedule = Schedule(n_clouds, n_fogs, fog_cloud_paths, n_tasks)

    # convert cloud schedule
    cloud_arg_min = np.argmin(cloud_matrix, axis=1)
    cloud_sorted_indices = np.argsort(cloud_matrix, axis=0).T
    for cloud_id in range(cloud_sorted_indices.shape[0]):
        for task_id, x in enumerate(cloud_arg_min):
            if cloud_id == x:
                schedule.cloud_schedule[cloud_id].append(task_id)

    # convert fog schedule
    fog_arg_min = np.argmin(fog_matrix, axis=1)
    fog_sorted_indices = np.argsort(fog_matrix, axis=0).T
    for fog_id in range(fog_sorted_indices.shape[0]):
        for task_id, x in enumerate(fog_arg_min):
            if fog_id == x:
                schedule.fog_schedule[fog_id].append(task_id)

    schedule.is_valid()

    return schedule


if __name__ == '__main__':
    clouds, fogs = load_cloudlets('../data/input_data/cloudlet_2_5.json')
    tasks = load_tasks('../data/input_data/tasks_10.json')

    fog_cloud_paths = np.array([fog.idle_cl_gamma for fog in fogs]) != float('inf')

    cloud_matrix = np.random.uniform(-1, 1, (len(tasks), len(clouds)))
    fog_matrix = np.random.uniform(-1, 1, (len(tasks), len(fogs)))

    schedule = matrix_to_schedule(cloud_matrix, fog_matrix, fog_cloud_paths)
    print(schedule)
    print(fog_cloud_paths)
    print(schedule.is_valid())
