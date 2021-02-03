#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:16, 03/02/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from config import Config, OptExp
from pandas import DataFrame, read_csv
from numpy import zeros, vstack
from pathlib import Path

n_timebound = "no_time_bound"
pathsave = f'{Config.RESULTS_DATA}/{n_timebound}/final_table.csv'

cols = ["Task", "Model", "ER-MEAN", "ER-CV", "GD-MEAN", "GD-CV", "IGD-MEAN", "IGD-CV",
        "STE-MEAN", "STE-CV", "HV-MEAN", "HV-CV", "HAR-MEAN", "HAR-CV"]

metrics_matrix = zeros((1, len(cols)))
for n_tasks in OptExp.N_TASKS:
    path_statistics_file = f'{Config.RESULTS_DATA}/{n_timebound}/task_{n_tasks}/{Config.METRICS}/metrics/statistics.csv'
    df = read_csv(path_statistics_file, usecols=cols)
    metrics_matrix = vstack((metrics_matrix, df.values))
metrics_matrix = metrics_matrix[1:]

df1 = DataFrame(metrics_matrix, columns=cols)
df1.to_csv(pathsave, index=False)

