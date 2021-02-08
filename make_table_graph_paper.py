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
from utils.visual.bar import group_bar2d

## Get the results and make the tables
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

## Draw some figures

## get the data for figures
data_types = ["MEAN"]
metrics = ["ER", "GD", "IGD", "STE", "HV", "HAR"]
groups = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
models = ["NSGA-II", "NSGA-III", "MO-ALO", "MO-SSA"]
pathsave = f'{Config.RESULTS_DATA}/{n_timebound}/paper/'
Path(pathsave).mkdir(parents=True, exist_ok=True)

for data_type in data_types:
    for metric in metrics:
        data = [df1[df1["Model"] == model][metric + "-" + data_type] for model in models]
        xy_labels = ["#Tasks", data_type]
        title = f"Metric: {metric}"
        filename = f"{data_type}-{metric}"
        group_bar2d(groups, data, models, xy_labels, title, [pathsave]*2, filename, [".png", ".pdf"], False)

