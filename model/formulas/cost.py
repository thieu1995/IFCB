#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:47, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from model.schedule import Schedule


def data_forwarding_cost(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    cloud_cost = 0
    fog_cost = 0
    peer_cost = 0

    tasks_fogs_schedule = {}
    for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
        for task_id in list_task_id:
            tasks_fogs_schedule[task_id] = fog_id

    for time_slot in range(schedule.total_time):
        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            fog = fogs[fog_id]
            fog_cost += fog.sigma_device_idle + fog.sigma_idle
            if len(list_task_id) > time_slot:
                task = tasks[list_task_id[time_slot]]
                fog_cost += (fog.eg_sigma + fog.fi_sigma) * (task.r_p + task.r_s)

        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            if len(list_task_id) <= time_slot:
                continue
            task_id = list_task_id[time_slot]
            task = tasks[task_id]
            fog = fogs[tasks_fogs_schedule[task_id]]
            cloud = clouds[cloud_id]
            cloud_cost += fog.sigma_device_idle + fog.sigma_idle + cloud.sigma_idle
            cloud_cost += (fog.sigma_device + fog.sigma + cloud.sigma) * (task.q_p + task.q_s)

    list_task_handlers = schedule.get_list_task_handlers()
    for peer_id, list_task_id in schedule.schedule_peers_tasks.item():
        for task_id in list_task_id:
            fog_id, cloud_id = list_task_handlers[task_id]
            fog = fogs[fog_id]
            cloud = clouds[cloud_id]
            peer = peers[peer_id]
            task = tasks[task_id]
            peer_cost += (fog.sigma_device_idle + fog.sigma_idle + peer.sigma_sm) + \
                          (fog.sigma_device + fog.sigma + peer.sigma) * (task.r_p + task.r_s) + \
                          (fog.sigma_device_idle + fog.sigma_idle + cloud.sigma_idle + peer.sigma_sm) + \
                          (fog.sigma_device + fog.sigma + cloud.sigma + peer.sigma) * (task.q_p + task.q_s)

    return cloud_cost + fog_cost + peer_cost


def computation_cost(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    cloud_cost = 0
    fog_cost = 0

    for time_slot in range(schedule.total_time):
        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            fog = fogs[fog_id]
            fog_cost += fog.pi_idle
            if len(list_task_id) > time_slot:
                task = tasks[list_task_id[time_slot]]
                fog_cost += fog.pi * task.r_p
                start_time_slot = max(0, time_slot - fog.tau)
                for i in range(start_time_slot, time_slot - 1):
                    task = tasks.values()[i]
                    factor = 1 / 2 ** (time_slot - i + 1)
                    fog_cost += factor * fog.pi * task.r_s

        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            cloud = clouds[cloud_id]
            cloud_cost += cloud.pi_idle
            if len(list_task_id) > time_slot:
                task_id = list_task_id[time_slot]
                task = tasks[task_id]
                cloud_cost += cloud.pi * task.q_p
                for i in range(time_slot - 1):
                    task = tasks[i]
                    factor = 1 / 2 ** (time_slot - i + 1)
                    cloud_cost += factor * cloud.pi * task.q_s

    return cloud_cost + fog_cost


def storage_cost(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    cloud_cost = 0
    fog_cost = 0
    peer_cost = 0

    for time_slot in range(schedule.total_time):
        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            cloud = clouds[cloud_id]
            cloud_cost += cloud.omega_idle
            for i in range(time_slot):
                if len(list_task_id) > i:
                    task = tasks[list_task_id[i]]
                    cloud_cost += cloud.omega * task.q_s

        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            fog = fogs[fog_id]
            fog_cost += fog.omega_idle
            start_time_slot = max(0, time_slot - fog.tau)
            for i in range(start_time_slot, time_slot):
                if len(list_task_id) > i:
                    task = tasks[list_task_id[i]]
                    fog_cost += fog.omega * task.r_s

    for peer_id, list_task_id in schedule.schedule_peers_tasks.items():
        for task_id in list_task_id:
            peer = peers[peer_id]
            task = tasks[task_id]
            peer_cost += peer.omega_sm + peer.omega * (task.r_s + task.r_p + task.q_s + task.q_p)

    return cloud_cost + fog_cost + peer_cost

