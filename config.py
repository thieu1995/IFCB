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
    MODE = 'epoch'  # epoch, fe (function evaluation counter instead of epoch)
    TIME_BOUND_KEY = False  # time bound for the training process
    TIME_BOUND_VALUE = 100

    METRICS_MAX = ["weighting-min", ]           # other methods need min - for calculate the global best fitness
    METRICS_NEED_MIN_OBJECTIVE_VALUES = False   # For tunning all parameter to find the min-objective value of each objective.
    MULTI_OBJECTIVE_SUPPORTERS = ["BaseNSGA", "BaseNSGA_II", "BaseNSGA_III", "BaseNSGA_C", "LSHADE", "BaseMO_SSA"]

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
    METRICS = 'pareto'
    OBJ_WEIGHTING_METRICS = [0.2, 0.3, 0.5]
    OBJ_DISTANCING_METRICS = [800, 40000, 500]  ## DEMAND-LEVEL REQUIREMENT
    OBJ_MINMAX_METRICS = [800, 40000, 500]
    OBJ_WEIGHTING_MIN_METRICS_1 = [0.2, 0.3, 0.5]
    OBJ_WEIGHTING_MIN_METRICS_2 = [800, 40000, 500]


class DefaultData:
    R_PROCESSING_BOUND = [50_000, 500_000]  # 100 KB - 1000 KB (0.1 - 1.0 MB)
    R_STORAGE_BOUND = [50_000, 500_000]
    Q_PROCESSING_BOUND = [50_000, 500_000]
    Q_STORAGE_BOUND = [50_000, 500_000]
    SERVICE_LATENCY_MAX = [10, 100]                 # 10 seconds to 100 seconds
    TASK_LIST = list(range(50, 1001, 50))

    TASK_LABEL_IMPORTANT = 1
    TASK_DEFAULT_SL_MAX = 10

    NUM_TASKS = 1000
    NUM_CLOUDS = 2
    NUM_FOGS = 8
    NUM_PEERS = 5

    LOC_LONG_BOUND = [-100, 100]
    LOC_LAT_BOUND = [-100, 100]

    RATE_FOG_CLOUD_LINKED = 1.0
    RATE_FOG_PEER_LINKED = 2
    RATE_CLOUD_PEER_LINKED = 2


class OptParas:     # Optimizer parameters config
    GA = {
        "p_c": [0.9],
        "p_m": [0.05]
    }
    PSO = {
        "w_min": [0.4],
        "w_max": [0.9],
        "c_local": [1.2],
        "c_global": [1.2]
    }
    WOA = {             # This parameters are actually fixed parameters in WOA
        "p": [0.5],
        "b": [1.0]
    }
    EO = {              # This parameters are actually fixed parameters in EO
        "V": [1.0],
        "a1": [2.0],
        "a2": [1.0],
        "GP": [0.5]
    }
    AEO = {             # This algorithm has no actually parameters
        "No": [None]
    }
    SSA = {
        "ST": [0.8],    # ST in [0.5, 1.0]
        "PD": [0.2],    # number of producers
        "SD": [0.1]     # number of sparrows who perceive the danger
    }

    ### Multi-objectives
    NSGA_II = {
        "p_c": [0.9],
        "p_m": [0.05]
    }
    NSGA_III = {
        "p_c": [0.9],
        "p_m": [0.05],
        "cof_divs": [16],
        "old_pop_rate": [0.7]
    }
    MO_SSA = {
        "ST": [0.8],
        "PD": [0.2],
        "SD": [0.1]
    }


class OptExp:       # Optimizer paras in experiments
    N_TRIALS = 1
    N_TASKS = [50]
    TIME_BOUND_VALUES = [60, 100]
    POP_SIZE = [50]
    LB = [-1]
    UB = [1]
    EPOCH = [5]
    FE = [100000]

