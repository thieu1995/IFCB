#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:48, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from model.schedule import Schedule


def transmission_latency(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    cloud_latency = 0
    fog_latency = 0

    tasks_fogs_schedule = {}
    for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
        for task_id in list_task_id:
            tasks_fogs_schedule[task_id] = fog_id

    for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
        for task_id in range(list_task_id):
            fog = fogs[tasks_fogs_schedule[task_id]]
            task = tasks[task_id]
            cloud = clouds[cloud_id]
            cloud_latency += (fog.eta + cloud.eta) * (task.q_p + task.q_s)

    for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
        for task_id in range(list_task_id):
            fog = fogs[fog_id]
            task = tasks[task_id]
            fog_latency += fog.eta * (task.r_p + task.r_s)

    return cloud_latency + fog_latency



def processing_latency(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    cloud_latency = 0
    fog_latency = 0

    for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
        cloud = clouds[cloud_id]
        for time_slot, task_id in enumerate(list_task_id):
            task = tasks[task_id]
            cloud_latency += cloud.lamda * task.q_p
            for i in range(time_slot - 1):
                task = task.values()[i]
                factor = 1 / 2 ** (time_slot - i + 1)
                cloud_latency += factor * cloud.lamda * task.q_s

    for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
        fog = fogs[fog_id]
        for time_slot, task_id in enumerate(list_task_id):
            task = tasks[task_id]
            fog_latency += fog.lamda * task.r_p
            start_time_slot = max(0, time_slot - fog.tau)
            for i in range(start_time_slot, time_slot - 1):
                task = tasks.values()[i]
                factor = 1 / 2 ** (time_slot - i + 1)
                fog_latency += factor * fog.lamda * task.r_s

    return cloud_latency + fog_latency

