#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:48, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from typing import List

from computable import Cloud, Fog, Task

from ..schedule import Schedule


def processing_latency(clouds: List[Cloud], fogs: List[Fog], tasks: List[Task], schedule: Schedule) -> float:
    cloud_latency = 0
    fog_latency = 0

    for cloud_id, cloud_node in enumerate(schedule.cloud_schedule):
        cloud = clouds[cloud_id]
        for time_slot, task_id in enumerate(cloud_node):
            task = tasks[task_id]
            cloud_latency += cloud.lamda * task.q_r
            for i in range(time_slot - 1):
                task = tasks[i]
                factor = 1 / 2 ** (time_slot - i + 1)
                cloud_latency += factor * cloud.lamda * task.q_s

    for fog_id, fog_node in enumerate(schedule.fog_schedule):
        fog = fogs[fog_id]
        for time_slot, task_id in enumerate(fog_node):
            task = tasks[task_id]
            fog_latency += fog.lamda * task.p_r
            start_time_slot = max(0, time_slot - fog.tau)
            for i in range(start_time_slot, time_slot - 1):
                task = tasks[i]
                factor = 1 / 2 ** (time_slot - i + 1)
                fog_latency += factor * fog.lamda * task.p_s

    return cloud_latency + fog_latency


def transmission_latency(clouds: List[Cloud], fogs: List[Fog], tasks: List[Task], schedule: Schedule) -> float:
    cloud_latency = 0
    fog_latency = 0

    inverted_fog_schedule = [None for _ in range(schedule.n_tasks)]
    for fog_id, fog in enumerate(schedule.fog_schedule):
        for task_id in fog:
            inverted_fog_schedule[task_id] = fog_id

    for cloud_id, cloud_node in enumerate(schedule.cloud_schedule):
        for task_id in range(cloud_node):
            fog = fogs[inverted_fog_schedule[task_id]]
            task = tasks[task_id]
            cloud_latency += (fog.ef_delta + fog.fg_delta[cloud_id]) * (task.q_r + task.q_s)

    for fog_id, fog_node in enumerate(schedule.fog_schedule):
        for task_id in range(fog_node):
            fog = fogs[fog_id]
            task = tasks[task_id]
            fog_latency += fog.ef_delta * (task.p_r + task.p_s)

    return cloud_latency + fog_latency
