#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:16, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import json
from pathlib import Path
from typing import List, Tuple

from model import Cloud, Fog, Task
from config import Config


def dump_cloudlets(clouds: List[Cloud], fogs: List[Fog]) -> None:
    data = {
        'fogs': [f.to_dict() for f in fogs],
        'clouds': [c.to_dict() for c in clouds],
    }

    output_file = Path(f'{Config.INPUT_DATA}/cloudlet_{len(clouds)}_{len(fogs)}.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open('w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2)


def dump_tasks(tasks: List[Task]) -> None:
    tasks = [t.to_dict() for t in tasks]

    output_file = Path(f'{Config.INPUT_DATA}/tasks_{len(tasks)}.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open('w', encoding='utf-8') as outfile:
        json.dump(tasks, outfile, indent=2)


def load_tasks(filename: str) -> List[Task]:
    tasks = []
    with open(filename, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        for row in data:
            task = Task()
            for key, value in row.items():
                setattr(task, key, value)
            tasks.append(task)
        return tasks


def load_cloudlets(filename: str) -> Tuple[List[Cloud], List[Fog]]:
    clouds = []
    fogs = []
    with open(filename, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        cloud_data = data['clouds']
        fog_data = data['fogs']
        for row in cloud_data:
            cloud = Cloud()
            for key, value in row.items():
                setattr(cloud, key, value)
            clouds.append(cloud)
        for row in fog_data:
            fog = Fog()
            for key, value in row.items():
                setattr(fog, key, value)
            fogs.append(fog)
        return clouds, fogs
