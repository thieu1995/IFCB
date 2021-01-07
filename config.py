#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:15, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    CORE_DATA_DIR = f'{basedir}/data'
    INPUT_DATA = f'{CORE_DATA_DIR}/input_data'
    RESULTS_DATA = f'{CORE_DATA_DIR}/final_results'
    METRICS = 'trade-off'  # power, latency, cost, trade-off
    MODE = 'time'  # time, epochs


class DefaultData:
    R_PROCESSING_BOUND = [100_000, 100_000_000]  # 0.1 MB - 10 MB
    R_STORAGE_BOUND = [100_000, 100_000_000]
    Q_PROCESSING_BOUND = [100_000, 100_000_000]
    Q_STORAGE_BOUND = [100_000, 100_000_000]
    SERVICE_LATENCY_MAX = [10, 100]                 # 10 seconds to 100 seconds
    TASK_LIST = list(range(100, 1001, 100))


    NUM_TASKS = 20
    NUM_FOGS = 10
    NUM_CLOUDS = 2
    NUM_PEERS = 5

    LOC_LONG_BOUND = [-100, 100]
    LOC_LAT_BOUND = [-100, 100]

    RATE_FOG_CLOUD_LINKED = 0.85
    RATE_FOG_PEER_LINKED = 0.2










