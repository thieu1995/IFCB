#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:15, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from os.path import abspath, dirname

basedir = abspath(dirname(__file__))


class Config:
    CORE_DATA_DIR = f'{basedir}/data'
    INPUT_DATA = f'{CORE_DATA_DIR}/input_data'
    RESULTS_DATA = f'{CORE_DATA_DIR}/final_results'
    MODE = 'time'  # time, epoch, fe (function evaluation counter instead of epoch)

    ### Single Objective
    # 1. power              --> find Min
    # 2. latency            --> find Min
    # 3. cost               --> find Min

    ### Multiple Objective
    ## Single target
    # 1. weighting          --> find Min
    # 2. distancing (demand-level vector)       --> find Min
    # 3. min-max formulation                    --> find Min
    # 4. weighting-min formulation  # the paper of Thang and Khiem      --> find Max

    ## Multi-target
    # 1. Pareto-front

    ## finally: metrics = ["power", "latency", "cost", "weighting", "distancing", "min-max", "weighting-min", "pareto",...]
    METRICS_NEED_MIN = False
    METRICS_MAX = ["weighting-min", ]
    METRICS = 'min-max'
    OBJ_WEIGHTING_METRICS = [0.2, 0.3, 0.5]
    OBJ_DISTANCING_METRICS = [800, 40000, 500]  ## DEMAND-LEVEL REQUIREMENT
    OBJ_MINMAX_METRICS = [800, 40000, 500]
    OBJ_WEIGHTING_MIN_METRICS_1 = [0.2, 0.3, 0.5]
    OBJ_WEIGHTING_MIN_METRICS_2 = [800, 40000, 500]


class DefaultData:
    R_PROCESSING_BOUND = [100_000, 100_000_000]  # 0.1 MB - 10 MB
    R_STORAGE_BOUND = [100_000, 100_000_000]
    Q_PROCESSING_BOUND = [100_000, 100_000_000]
    Q_STORAGE_BOUND = [100_000, 100_000_000]
    SERVICE_LATENCY_MAX = [10, 100]                 # 10 seconds to 100 seconds
    TASK_LIST = list(range(100, 1001, 100))

    TASK_LABEL_IMPORTANT = 1
    TASK_DEFAULT_SL_MAX = 10

    NUM_TASKS = 20
    NUM_FOGS = 10
    NUM_CLOUDS = 2
    NUM_PEERS = 5

    LOC_LONG_BOUND = [-100, 100]
    LOC_LAT_BOUND = [-100, 100]

    RATE_FOG_CLOUD_LINKED = 0.8
    RATE_FOG_PEER_LINKED = 0.4
    RATE_CLOUD_PEER_LINKED = 0.2










