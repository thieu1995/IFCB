#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:56, 22/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import all, any, zeros, ndarray, min, sqrt, mean, sum, max, abs
from numpy.linalg import norm
from pygmo.core import hypervolume

## GD, IGD, STE: A Comparative Study of Recent Multi-objective Metaheuristics for Solving Constrained Truss Optimisation Problems
## ER: Multi-objective particle swarm optimization with random immigrants

## Multi-objectiveEvolutionaryOptimisationforProductDesignandManufacturing.pdf
##
##  Metrics for evaluating convergence: error ratio, distance from reference set,...
##  Metrics for evaluating spread of solutions: spread, spacing,...
##  Metrics for evaluating certain combinations of convergence and spread of solutions to the known Pareto-optimal front
##      including: hypervolume, coverage, R-metrics,...

## On metrics for comparing nondominated sets.

## Performance Metrics Ensemble for Multiobjective Evolutionary Algorithms

def dominates(fit_a, fit_b):
    return all(fit_a <= fit_b) and any(fit_a < fit_b)


def find_dominates_list(list_fit: ndarray):
    size = len(list_fit)
    list_dominated = zeros(size)  # 0: non-dominated, 1: dominated by someone
    for i in range(0, size):
        list_dominated[i] = 0
        for j in range(0, i):
            if any(list_fit[i] != list_fit[j]):
                if dominates(list_fit[i], list_fit[j]):
                    list_dominated[j] = 1
                elif dominates(list_fit[j], list_fit[i]):
                    list_dominated[i] = 1
                    break
            else:
                list_dominated[j] = 1
                list_dominated[i] = 1
    return list_dominated


def generational_distance(pareto_fronts: ndarray, reference_fronts:ndarray):
    size_refs = len(reference_fronts)
    size_pars = len(pareto_fronts)
    gd = 0
    for i in range(size_pars):
        dist_min = min([norm(pareto_fronts[i] - reference_fronts[j]) for j in range(0, size_refs)])
        gd += dist_min**2
    return sqrt(gd) / size_pars


def inverted_generational_distance(pareto_fronts: ndarray, reference_fronts: ndarray):
    size_refs = len(reference_fronts)
    size_pars = len(pareto_fronts)
    igd = 0
    for i in range(size_refs):
        dist_min = min([norm(reference_fronts[i] - point) for point in pareto_fronts])
        igd += dist_min ** 2
    return sqrt(igd) / size_refs


def error_ratio(pareto_fronts: ndarray, reference_fronts: ndarray):
    # Multi-objective particle swarm optimization with random immigrants
    count = 0
    for point in pareto_fronts:
        list_flags = [all(point == solution) for solution in reference_fronts]
        if not any(list_flags):
            count += 1
    return count/len(reference_fronts)


def spacing_to_extent(pareto_fronts: ndarray):
    size_pars = len(pareto_fronts)

    dist_min_list = zeros(size_pars)
    for i in range(size_pars):
        dist_min = min([norm(pareto_fronts[i] - pareto_fronts[j]) for j in range(size_pars) if i != j])
        dist_min_list[i] = dist_min
    dist_mean = mean(dist_min_list)
    spacing = sum((dist_min_list - dist_mean) ** 2) / (size_pars - 1)

    f_max = max(pareto_fronts, axis=0)
    f_min = min(pareto_fronts, axis=0)
    extent = sum(abs(f_max - f_min))

    ste = spacing / extent
    return ste


def hyper_volume(pareto_fronts:ndarray, reference_fronts:ndarray, hv_worst_point=None, hv_point=10000):
    if hv_worst_point is None:
        print("Need HV worst point")
        exit(0)
    hv_pf = hypervolume(pareto_fronts)
    return hv_pf.compute(hv_worst_point + hv_point)


def hyper_area_ratio(pareto_fronts: ndarray, reference_fronts: ndarray, hv_worst_point=None, hv_point=10000):
    if hv_worst_point is None:
        print("Need HV worst point")
        exit(0)
    hv_pf = hypervolume(pareto_fronts)
    hv_rf = hypervolume(reference_fronts)
    hv_point = hv_worst_point + hv_point
    return hv_pf.compute(hv_point) / hv_rf.compute(hv_point)








