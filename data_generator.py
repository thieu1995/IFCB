#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:18, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import random

from model import Cloud, Fog, Task
from utils.io_util import dump_cloudlets, dump_tasks


def create_cloud():
    cloud = Cloud()

    cloud.idle_beta = random.uniform(50, 200)  # power consumption - computation
    cloud.beta = random.uniform(1e-9, 1e-6)  # power consumption - computation

    cloud.idle_alpha = random.uniform(30, 200)  # power consumption - storage
    cloud.alpha = random.uniform(5e-9, 5e-6)  # power consumption - storage

    cloud.lam_bda = random.uniform(1e-8, 1e-7)  # latency - processing

    cloud.idle_pi = random.uniform(3.6e-7, 1.1e-6)  # cost - computation
    cloud.pi = random.uniform(3.6e-15, 3.6e-14)  # cost - computation

    cloud.idle_omega = random.uniform(2e-8, 4e-8)  # cost - storage
    cloud.omega = random.uniform(2e-16, 2e-15)  # cost - storage

    return cloud


def create_fog(number_clouds):
    fog = Fog()
    idle_beta_range = [100, 500]  # power for computation idle
    beta_range = [5e-7, 5e-4]  # power for computation

    idle_alpha_range = [10, 100]  # power for storage idle
    alpha_range = [5e-8, 5e-5]  # power for storage

    idle_eg_gamma_range = [50, 200]  # power for data forwarding at edge gateway idle
    eg_gamma_range = [5e-8, 5e-5]  # power for data forwarding at edge gateway

    idle_fi_gamma_range = [50, 200]  # power for data forwarding at fog instance idle
    fi_gammma_range = [5e-8, 5e-5]  # power for data forwarding at fog instance

    lam_bda_range = [1e-8, 1e-7]  # processing latency

    idle_pi_range = [1.8e-7, 5.5e-7]  # cost
    pi_range = [1.8e-15, 1.8e-14]  # cost

    idle_omega_range = [1e-8, 2e-8]  # cost
    omega_range = [1e-16, 1e-15]  # cost

    ef_delta_range = [0.005, 0.05]  # tranmission latency

    idle_eg_sigma_range = [0.001, 0.008]  # cost data forwarding
    eg_sigma_range = [5e-10, 5e-9]  # cost data forwarding

    idle_fi_sigma_range = [0.001, 0.01]  # cost data forwarding
    fi_sigma_range = [5e-10, 5e-9]  # cost data forwarding

    tau = random.randint(0, 20)
    # tau = 0

    idle_cl_gamma_range = [50, 75]  # power cloud
    cl_gamma_range = [5e-7, 5e-6]  # power cloud

    fg_delta_range = [0.2, 5.0]  # latency

    idle_cl_sigma_range = [0.02, 0.1]  # cost
    cl_sigma_range = [5e-9, 5e-8]  # cost

    fog.idle_beta = random.uniform(idle_beta_range[0], idle_beta_range[1])  # power for computation idle
    fog.beta = random.uniform(beta_range[0], beta_range[1])  # power for computation

    fog.idle_alpha = random.uniform(idle_alpha_range[0], idle_alpha_range[1])  # power for storage idle
    fog.alpha = random.uniform(alpha_range[0], alpha_range[1])  # power for storage

    fog.idle_eg_gamma = random.uniform(idle_eg_gamma_range[0], idle_eg_gamma_range[1])  # power for data forwarding at edge gateway idle
    fog.eg_gamma = random.uniform(eg_gamma_range[0], eg_gamma_range[1])  # power for data forwarding at edge gateway

    fog.idle_fi_gamma = random.uniform(idle_fi_gamma_range[0], idle_fi_gamma_range[1])  # power for data forwarding at fog instance idle
    fog.fi_gammma = random.uniform(fi_gammma_range[0], fi_gammma_range[1])  # power for data forwarding at fog instance

    fog.idle_cl_gamma = [float('inf') for _ in range(number_clouds)]  # power for data forwarding at cloud nodes idle
    fog.cl_gamma = [float('inf') for _ in range(number_clouds)]  # power for data forwarding at cloud nodes

    fog.lam_bda = random.uniform(lam_bda_range[0], lam_bda_range[1])

    fog.idle_pi = random.uniform(idle_pi_range[0], idle_pi_range[1])
    fog.pi = random.uniform(pi_range[0], pi_range[1])

    fog.idle_omega = random.uniform(idle_omega_range[0], idle_omega_range[1])
    fog.omega = random.uniform(omega_range[0], omega_range[1])

    fog.ef_delta = random.uniform(ef_delta_range[0], ef_delta_range[1])
    fog.fg_delta = [float('inf') for _ in range(number_clouds)]

    fog.idle_eg_sigma = random.uniform(idle_eg_sigma_range[0], idle_eg_sigma_range[1])
    fog.eg_sigma = random.uniform(eg_sigma_range[0], eg_sigma_range[1])

    fog.idle_fi_sigma = random.uniform(idle_fi_sigma_range[0], idle_fi_sigma_range[1])
    fog.fi_sigma = random.uniform(fi_sigma_range[0], fi_sigma_range[1])

    fog.idle_cl_sigma = [float('inf') for _ in range(number_clouds)]
    fog.cl_sigma = [float('inf') for _ in range(number_clouds)]

    fog.tau = random.randint(0, 20)

    # ensure each fog always connects at least a cloud
    i = random.randrange(number_clouds)
    fog.idle_cl_gamma[i] = random.uniform(idle_cl_gamma_range[0], idle_cl_gamma_range[1])
    fog.cl_gamma[i] = random.uniform(cl_gamma_range[0], cl_gamma_range[1])

    fog.fg_delta[i] = random.uniform(fg_delta_range[0], fg_delta_range[1])

    fog.idle_cl_sigma[i] = random.uniform(idle_cl_sigma_range[0], idle_cl_sigma_range[1])
    fog.cl_sigma[i] = random.uniform(cl_sigma_range[0], cl_sigma_range[1])

    for i in range(number_clouds):
        if random.random() < 1:
            fog.idle_cl_gamma[i] = random.uniform(idle_cl_gamma_range[0], idle_cl_gamma_range[1])
            fog.cl_gamma[i] = random.uniform(cl_gamma_range[0], cl_gamma_range[1])

            fog.fg_delta[i] = random.uniform(fg_delta_range[0], fg_delta_range[1])

            fog.idle_cl_sigma[i] = random.uniform(idle_cl_sigma_range[0], idle_cl_sigma_range[1])
            fog.cl_sigma[i] = random.uniform(cl_sigma_range[0], cl_sigma_range[1])

    return fog


def create_task():
    task = Task()
    task.p_r = random.randint(100_000, 100_000_000)  # 0.1 MB - 10 MB
    task.p_s = random.randint(100_000, 100_000_000)
    task.q_r = random.randint(100_000, 100_000_000)
    task.q_s = random.randint(100_000, 100_000_000)
    return task


if __name__ == '__main__':
    random.seed(None)

    # number_clouds = input('Number clouds: ')
    # number_fogs = input('Number fogs: ')
    number_clouds = 5
    number_fogs = 20
    # number_tasks = input('Number tasks: ')
    number_tasks = range(50, 501, 50)
    # print(number_tasks)

    clouds = [create_cloud() for _ in range(int(number_clouds))]
    fogs = [create_fog(int(number_clouds)) for _ in range(int(number_fogs))]
    dump_cloudlets(clouds, fogs)
    for _number_tasks in number_tasks:
        tasks = [create_task() for _ in range(int(_number_tasks))]
        dump_tasks(tasks)

