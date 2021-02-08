#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:48, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from model.schedule import Schedule


def data_forwarding_power(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    fog_power = 0
    cloud_power = 0
    peer_power = 0

    tasks_fogs_schedule = {}
    for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
        for task_id in list_task_id:
            tasks_fogs_schedule[task_id] = fog_id
    tasks_clouds_schedule = {}
    for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
        for task_id in list_task_id:
            tasks_clouds_schedule[task_id] = cloud_id

    for time_slot in range(schedule.total_time):
        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            fog = fogs[fog_id]
            fog_power += fog.alpha_device_idle + fog.alpha_idle
            if len(list_task_id) > time_slot:
                task = tasks[list_task_id[time_slot]]
                fog_power += (fog.alpha_device + fog.alpha) * (task.r_p + task.r_s)

        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            if len(list_task_id) <= time_slot:
                continue
            task_id = list_task_id[time_slot]
            task = tasks[task_id]
            fog = fogs[tasks_fogs_schedule[task_id]]
            cloud = clouds[cloud_id]
            cloud_power += fog.alpha_device_idle + fog.alpha_idle + cloud.alpha_idle
            cloud_power += (fog.alpha_device + fog.alpha + cloud.alpha) * (task.q_p + task.q_s)

    list_task_handlers = schedule.get_list_task_handlers()
    for peer_id, list_task_id in schedule.schedule_peers_tasks.items():
        for task_id in list_task_id:
            fog_id, cloud_id = list_task_handlers[task_id]
            fog = fogs[fog_id]
            cloud = clouds[cloud_id]
            peer = peers[peer_id]
            task = tasks[task_id]
            peer_power += (fog.alpha_device_idle + fog.alpha_idle + peer.alpha_sm) + \
                         (fog.alpha_device + fog.alpha + peer.alpha) * (task.r_p + task.r_s) + \
                          (fog.alpha_device_idle + fog.alpha_idle + cloud.alpha_idle + peer.alpha_sm) + \
                          (fog.alpha_device + fog.alpha + cloud.alpha + peer.alpha) * (task.q_p + task.q_s)
    return fog_power + cloud_power + peer_power


def computation_power(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    fog_power = 0
    cloud_power = 0

    for time_slot in range(schedule.total_time):
        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            fog = fogs[fog_id]
            fog_power += fog.beta_idle
            if len(list_task_id) > time_slot:
                task = tasks[list_task_id[time_slot]]
                fog_power += fog.beta * task.r_p
                start_time_slot = max(0, time_slot - fog.tau)
                for j in range(start_time_slot, time_slot):
                    task = tasks[list_task_id[j]]
                    factor = 1 / ((time_slot - j) ** 2 + 1)
                    fog_power += factor * fog.beta * task.r_s

        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            cloud = clouds[cloud_id]
            cloud_power += cloud.beta_idle
            if len(list_task_id) > time_slot:
                task = tasks[list_task_id[time_slot]]
                cloud_power += cloud.beta * task.q_p
                for j in range(time_slot):
                    task = tasks[list_task_id[j]]
                    factor = 1 / ((time_slot - j) ** 2 + 1)
                    cloud_power += factor * cloud.beta * task.q_s

    return fog_power + cloud_power


def storage_power(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    cloud_power = 0
    fog_power = 0
    peer_power = 0

    for time_slot in range(schedule.total_time):
        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            cloud = clouds[cloud_id]
            cloud_power += cloud.gamma_idle
            for j in range(time_slot):
                if len(list_task_id) > j:
                    task = tasks[list_task_id[j]]
                    cloud_power += cloud.gamma * task.r_s

        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            fog = fogs[fog_id]
            fog_power += fog.gamma_idle
            start_time_slot = max(0, time_slot - fog.tau)
            for j in range(start_time_slot, time_slot):
                if len(list_task_id) > j:
                    task = tasks[list_task_id[j]]
                    fog_power += fog.gamma * task.q_s

    for peer_id, list_task_id in schedule.schedule_peers_tasks.items():
        for task_id in list_task_id:
            peer = peers[peer_id]
            task = tasks[task_id]
            peer_power += peer.gamma_sm + peer.gamma * (task.r_s + task.r_p + task.q_s + task.q_p)

    return fog_power + cloud_power + peer_power

