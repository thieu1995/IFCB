#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:23, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import ndarray, argmin
from model.schedule import Schedule
from config import DefaultData


def make_dict_from_list_object(list_obj):
    my_dict = {}
    for obj in list_obj:
        my_dict[obj.id] = obj
    return my_dict


def matrix_to_schedule(problem:dict, solution: ndarray) -> Schedule:
    """
    Convert matrix data to schedule object
    :param solution:
        [ n_clouds + n_fogs, t_tasks ]
    :return: Schedule obj or None
    """
    clouds = problem["clouds"]
    fogs = problem["fogs"]
    tasks = problem["tasks"]

    fogs_dict = make_dict_from_list_object(fogs)
    clouds_dict = make_dict_from_list_object(clouds)
    tasks_dict = make_dict_from_list_object(tasks)

    n_clouds = problem["n_clouds"]
    n_fogs = problem["n_fogs"]
    n_tasks = problem["n_tasks"]
    n_peers = problem["n_peers"]

    schedule = Schedule(problem)
    matrix_cloud, matrix_fog = solution[:n_clouds,], solution[n_clouds:,]

    # convert matrix_cloud to schedule.schedule_clouds_tasks
    list_task_stt = argmin(matrix_cloud, axis=0)  # List of [task-stt: cloud-stt] (not task_id)
    for cl_stt in range(n_clouds):
        list_task_id = []
        for task_stt, cloud_stt in enumerate(list_task_stt):
            if cl_stt == cloud_stt:
                list_task_id.append(tasks[task_stt].id)
        schedule.schedule_clouds_tasks[clouds[cl_stt].id] = list_task_id

    # convert matrix_fog to schedule.schedule_flogs_tasks
    list_task_stt = argmin(matrix_fog, axis=0)
    for fg_stt in range(n_fogs):
        list_task_id = []
        for task_stt, fog_stt in enumerate(list_task_stt):
            if fg_stt == fog_stt:
                list_task_id.append(tasks[task_stt].id)
        schedule.schedule_fogs_tasks[fogs[fg_stt].id] = list_task_id

    # create a schedule for blockchain peers
    task_peers = {}     # task_id: list[Peer]
    for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
        fog = fogs_dict[fog_id]
        list_peers = fog.linked_peers
        for task_id in list_task_id:
            task_peers[task_id] = list_peers

    for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
        cloud = clouds_dict[cloud_id]
        list_peers = cloud.linked_peers
        for task_id in list_task_id:
            if task_id in task_peers.keys():
                task_peers[task_id] = list(set(task_peers[task_id] + list_peers))

    ### Remove task which not saved in blockchain
    task_peers_important = {}
    for task_id, peers in task_peers.items():
        if tasks_dict[task_id].label == DefaultData.TASK_LABEL_IMPORTANT:
            task_peers_important[task_id] = peers
    ###
    peer_tasks = {}
    for task_id, list_peer_id in task_peers_important.items():
        for peer_id in list_peer_id:
            if peer_id in peer_tasks:
                peer_tasks[peer_id].append(task_id)
            else:
                peer_tasks[peer_id] = [task_id]
    schedule.schedule_peers_tasks = peer_tasks
    return schedule


# clouds, fogs = load_cloudlets('data/cloudlet_5_20.json')
# tasks = load_tasks('data/tasks_1000.json')
#
# fog_cloud_paths = np.array([fog.idle_cl_gamma for fog in fogs]) != float('inf')


# if __name__ == '__main__':
#     clouds, fogs = load_cloudlets('../data/input_data/cloudlet_2_5.json')
#     tasks = load_tasks('../data/input_data/tasks_10.json')
#
#     fog_cloud_paths = array([fog.idle_cl_gamma for fog in fogs]) != float('inf')
#
#     cloud_matrix = uniform(-1, 1, (len(tasks), len(clouds)))
#     fog_matrix = uniform(-1, 1, (len(tasks), len(fogs)))
#
#     schedule = matrix_to_schedule(cloud_matrix, fog_matrix, fog_cloud_paths)
#     print(schedule)
#     print(fog_cloud_paths)
#     print(schedule.is_valid())
#

