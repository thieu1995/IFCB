#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:47, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from typing import List

from computable import Cloud, Fog, Task

from ..schedule import Schedule


def data_forwarding_cost(clouds: List[Cloud], fogs: List[Fog], tasks: List[Task], schedule: Schedule) -> float:
    cloud_cost = 0
    fog_cost = 0

    inverted_fog_schedule = [None for _ in range(schedule.n_tasks)]
    for fog_id, fog in enumerate(schedule.fog_schedule):
        for task_id in fog:
            inverted_fog_schedule[task_id] = fog_id

    for time_slot in range(schedule.total_time):
        for fog_id, fog_node in enumerate(schedule.fog_schedule):
            fog = fogs[fog_id]
            fog_cost += fog.idle_eg_sigma + fog.idle_fi_sigma
            if len(fog_node) > time_slot:
                task = tasks[fog_node[time_slot]]
                fog_cost += (fog.eg_sigma + fog.fi_sigma) * (task.p_r + task.p_s)

        for cloud_id, cloud_node in enumerate(schedule.cloud_schedule):
            if len(cloud_node) <= time_slot:
                continue
            task_id = cloud_node[time_slot]
            task = tasks[task_id]
            fog = fogs[inverted_fog_schedule[task_id]]
            cloud_cost += fog.idle_eg_sigma + fog.idle_fi_sigma + fog.idle_cl_sigma[cloud_id]
            cloud_cost += (fog.eg_sigma + fog.fi_sigma + fog.cl_sigma[cloud_id]) * (task.q_r + task.q_s)

    return cloud_cost + fog_cost


def computation_cost(clouds: List[Cloud], fogs: List[Fog], tasks: List[Task], schedule: Schedule) -> float:
    cloud_cost = 0
    fog_cost = 0

    for time_slot in range(schedule.total_time):
        for fog_id, fog_node in enumerate(schedule.fog_schedule):
            fog = fogs[fog_id]
            fog_cost += fog.idle_pi
            if len(fog_node) > time_slot:
                task_id = fog_node[time_slot]
                task = tasks[task_id]
                fog_cost += fog.pi * task.p_r
                start_time_slot = max(0, time_slot - fog.tau)
                for i in range(start_time_slot, time_slot - 1):
                    task = tasks[i]
                    factor = 1 / 2 ** (time_slot - i + 1)
                    fog_cost += factor * fog.pi * task.p_s

        for cloud_id, cloud_node in enumerate(schedule.cloud_schedule):
            cloud = clouds[cloud_id]
            cloud_cost += cloud.idle_pi
            if len(cloud_node) > time_slot:
                task_id = cloud_node[time_slot]
                task = tasks[task_id]
                cloud_cost += cloud.pi * task.q_r
                for i in range(time_slot - 1):
                    task = tasks[i]
                    factor = 1 / 2 ** (time_slot - i + 1)
                    cloud_cost += factor * cloud.pi * task.q_s

    return cloud_cost + fog_cost


def storage_cost(clouds: List[Cloud], fogs: List[Fog], tasks: List[Task], schedule: Schedule) -> float:
    cloud_cost = 0
    fog_cost = 0

    for time_slot in range(schedule.total_time):
        for cloud_id, cloud_node in enumerate(schedule.cloud_schedule):
            cloud = clouds[cloud_id]
            cloud_cost += cloud.idle_omega
            for i in range(time_slot):
                if len(cloud_node) > i:
                    task = tasks[cloud_node[i]]
                    cloud_cost += cloud.omega * task.q_s

        for fog_id, fog_node in enumerate(schedule.fog_schedule):
            fog = fogs[fog_id]
            fog_cost += fog.idle_omega
            start_time_slot = max(0, time_slot - fog.tau)
            for i in range(start_time_slot, time_slot):
                if len(fog_node) > i:
                    task = tasks[fog_node[i]]
                    fog_cost += fog.omega * task.p_s

    return cloud_cost + fog_cost

