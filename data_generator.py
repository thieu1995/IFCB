#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:18, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import randint, uniform, seed
from numpy import ceil
from model.fog import Fog
from model.cloud import Cloud
from model.task import Task
from model.blockchain.node import Node
from utils.io_util import dump_tasks, dump_nodes
from config import DefaultData


def create_task():
    r_p = randint(DefaultData.R_PROCESSING_BOUND[0], DefaultData.R_PROCESSING_BOUND[1])
    r_s = randint(DefaultData.R_STORAGE_BOUND[0], DefaultData.R_STORAGE_BOUND[1])
    q_p = randint(DefaultData.Q_PROCESSING_BOUND[0], DefaultData.Q_PROCESSING_BOUND[1])
    q_s = randint(DefaultData.Q_STORAGE_BOUND[0], DefaultData.Q_STORAGE_BOUND[1])
    label = 0 if uniform() < 0.5 else 1
    sl_max = randint(DefaultData.SERVICE_LATENCY_MAX[0], DefaultData.SERVICE_LATENCY_MAX[1])
    return Task(r_p, r_s, q_p, q_s, label, sl_max)


def create_cloud_node(name, location):
    cloud = Cloud(name, location)

    cloud.alpha = uniform(5e-7, 5e-5)     # power consumption - data forwarding
    cloud.beta = uniform(5e-9, 5e-7)      # power consumption - computation
    cloud.gamma = uniform(5e-9, 5e-6)     # power consumption - storage

    cloud.eta = uniform(0.2, 5.0)         # latency - transmission
    cloud.lamda = uniform(10e-8, 10e-7)   # latency - processing

    cloud.sigma = uniform(5e-9, 5e-8)     # cost - data forwarding
    cloud.pi = uniform(5e-15, 5e-14)      # cost - computation
    cloud.omega = uniform(5e-16, 5e-15)   # cost - storage

    cloud.alpha_idle = uniform(50, 200)   # power consumption - data forwarding - idle
    cloud.beta_idle = uniform(100, 200)   # power consumption - computation - idle
    cloud.gamma_idle = uniform(50, 200)   # power consumption - storage - idle

    cloud.sigma_idle = uniform(0.002, 0.01)   # cost - data forwarding - idle
    cloud.pi_idle = uniform(5e-7, 5e-6)       # cost - computation - idle
    cloud.omega_idle = uniform(2e-8, 5e-8)    # cost - storage - idle

    return cloud


def create_fog_node(name, location):
    fog = Fog(name, location)

    fog.alpha = uniform(5e-8, 5e-6)     # power consumption - data forwarding
    fog.beta = uniform(5e-7, 5e-4)      # power consumption - computation
    fog.gamma = uniform(5e-8, 5e-6)     # power consumption - storage

    fog.eta = uniform(0.005, 0.05)      # latency - transmission
    fog.lamda = uniform(10e-7, 10e-6)   # latency - processing

    fog.sigma = uniform(5e-10, 5e-9)    # cost - data forwarding
    fog.pi = uniform(2e-15, 2e-14)      # cost - computation
    fog.omega = uniform(10e-16, 10e-15) # cost - storage

    fog.alpha_idle = uniform(25, 75)    # power consumption - data forwarding - idle
    fog.beta_idle = uniform(100, 500)   # power consumption - computation - idle
    fog.gamma_idle = uniform(10, 100)   # power consumption - storage - idle

    fog.sigma_idle = uniform(0.001, 0.01)   # cost - data forwarding - idle
    fog.pi_idle = uniform(2e-7, 5e-7)       # cost - computation - idle
    fog.omega_idle = uniform(10e-8, 20e-8)  # cost - storage - idle

    fog.alpha_device = uniform(5e-8, 5e-5)          # power consumption - data forwarding - end device to fog
    fog.alpha_device_idle = uniform(50, 200)        # power consumption - data forwarding

    fog.sigma_device = uniform(5e-10, 5e-9)         # cost - data forwarding - end device to fog
    fog.sigma_device_idle = uniform(0.001, 0.008)   # cost - data forwarding - idle state

    fog.tau = randint(5, 20)
    return fog


def create_blockchain_node(name, location):
    node = Node(name, location)

    node.alpha = uniform(5e-10, 5e-9)       # power consumption - data forwarding
    node.gamma = uniform(5e-10, 5e-8)       # power consumption - storage

    node.sigma = uniform(5e-10, 5e-8)       # cost - data forwarding
    node.omega = uniform(10e-18, 10e-16)    # cost - storage

    node.alpha_sm = uniform(100, 250)       # power consumption - data forwarding - idle
    node.gamma_sm = uniform(50, 200)        # power consumption - storage - idle

    node.sigma_sm = uniform(0.0001, 0.001)  # cost - data forwarding - idle
    node.omega_sm = uniform(10e-10, 10e-8)  # cost - storage - idle

    return node


if __name__ == '__main__':
    seed(11)
    number_tasks = DefaultData.TASK_LIST
    for task in number_tasks:
        tasks = [create_task() for _ in range(int(task))]
        dump_tasks(tasks)

    number_fogs = DefaultData.NUM_FOGS
    number_clouds = DefaultData.NUM_CLOUDS
    number_peers = DefaultData.NUM_PEERS

    clouds = []
    for idx in range(number_clouds):
        name = "Cloud Node: " + str(idx)
        location = {
            "long": uniform(DefaultData.LOC_LONG_BOUND[0], DefaultData.LOC_LONG_BOUND[1]),
            "lat": uniform(DefaultData.LOC_LAT_BOUND[0], DefaultData.LOC_LAT_BOUND[1]),
        }
        clouds.append(create_cloud_node(name, location))

    fogs = []
    for idx in range(number_fogs):
        name = "Fog Node: " + str(idx)
        location = {
            "long": uniform(DefaultData.LOC_LONG_BOUND[0], DefaultData.LOC_LONG_BOUND[1]),
            "lat": uniform(DefaultData.LOC_LAT_BOUND[0], DefaultData.LOC_LAT_BOUND[1]),
        }
        fogs.append(create_fog_node(name, location))

    peers = []
    for idx in range(number_peers):
        name = "Peer Node: " + str(idx)
        location = {
            "long": uniform(DefaultData.LOC_LONG_BOUND[0], DefaultData.LOC_LONG_BOUND[1]),
            "lat": uniform(DefaultData.LOC_LAT_BOUND[0], DefaultData.LOC_LAT_BOUND[1]),
        }
        peers.append(create_blockchain_node(name, location))

    ## Connecting fog and cloud, fog and blockchain
    for id_fog, fog in enumerate(fogs):
        dist_list = []
        for id_cloud, cloud in enumerate(clouds):
            dist_temp = cloud.dist(fog)
            dist_list.append({"cloud_id": cloud.id, "dist": dist_temp})
        dist_list = sorted(dist_list, key=lambda item: item["dist"])
        dist_list = [item["cloud_id"] for item in dist_list]
        fog.linked_clouds = dist_list[:int(ceil(DefaultData.RATE_FOG_CLOUD_LINKED * number_clouds))]

        dist_list = []
        for id_peer, peer in enumerate(peers):
            dist_temp = peer.dist(fog)
            dist_list.append({"peer_id": peer.id, "dist": dist_temp})
        dist_list = sorted(dist_list, key=lambda item: item["dist"])
        dist_list = [item["peer_id"] for item in dist_list]
        fog.linked_peers = dist_list[:int(ceil(DefaultData.RATE_FOG_PEER_LINKED * number_peers))]

    ## Saving fog/cloud nodes and blockchain peers
    dump_nodes(clouds, fogs, peers)
