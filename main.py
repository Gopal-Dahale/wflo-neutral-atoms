# pylint: disable-all

import argparse
import json
import math
import time
from functools import partial

import dimod
import numpy as np
from hybrid import (
    ArgMin,
    Const,
    EnergyImpactDecomposer,
    InterruptableTabuSampler,
    Log,
    Loop,
    Race,
    SimulatedAnnealingSubproblemSampler,
    SplatComposer,
    State,
    TabuSubproblemSampler,
    TrackMin,
    min_sample,
)

from pasqal_hybrid import PasqalQAASubproblemSampler, PasqalQAOASubproblemSampler


def abs_dist(x, y):
    return abs(x - y)


# Checking and setting wake influence between pairs of cells
def check_wake(p_i, p_j):
    alpha = 0.09437
    rotor_radius = 27.881

    if p_j[1] > p_i[1]:
        # xdistance = abs_dist(p_i[0], p_j[0])
        ydistance = abs_dist(p_i[1], p_j[1])

        radius = rotor_radius + (alpha * ydistance)
        xmin = p_j[0] - radius
        xmax = p_j[0] + radius
        # print(xmin < p_i[0], xmax > p_i[0])
    else:
        # print("a", "b")
        return False

    if xmin < p_i[0] and xmax > p_i[0]:
        return True
    else:
        return False


# Calculating velocity factor between two cells due to wakes
def calc_velocity(loc_y_i, loc_y_j, u_inf):
    alpha = 0.09437
    a = 0.326795
    rotor_radius = 27.881

    # todo: check proper calculation of wind
    ydistance = abs_dist(
        loc_y_i, loc_y_j
    )  # euclidean_distance(0,0,loc_y_i, loc_y_j) #abs_dist(loc_y_i, loc_y_j)
    denom = pow((alpha * ydistance / rotor_radius) + 1, 2)
    # print(denom)
    return u_inf * (1 - (2 * a / denom))


# setting cell locations:
def calc_location(axis_, step):
    x = np.arange(0, axis_ * step, step)
    location_y, location_x = np.meshgrid(x, x)
    return location_x.flatten(), location_y.flatten()


def set_wind_regime(n_wind):
    if n_wind == 36:
        wind_speeds = np.array([8] * n_wind)
        wind_speeds = np.append(wind_speeds, [12] * n_wind)
        wind_speeds = np.append(wind_speeds, [17] * n_wind)

        prob_l = np.array([float(0.0047)] * n_wind)
        prob_l = np.append(prob_l, [0.008] * (n_wind - 9))  # 9 special regimes
        prob_l = np.append(
            prob_l, [0.01, 0.012, 0.0145, 0.014, 0.019, 0.014, 0.0145, 0.012, 0.01]
        )
        prob_l = np.append(prob_l, [0.011] * (n_wind - 9))
        prob_l = np.append(
            prob_l, [0.013, 0.017, 0.0185, 0.03, 0.035, 0.03, 0.0185, 0.017, 0.013]
        )
    elif n_wind == 1:
        wind_speeds = np.array([12] * n_wind)
        prob_l = np.array([1])

    return wind_speeds, prob_l


def check_wake_mat(p_i, p_j):
    alpha = 0.09437
    rotor_radius = 27.881

    if p_j[1] > p_i[1]:
        # xdistance = abs_dist(p_i[0], p_j[0])
        ydistance = abs_dist(p_i[1], p_j[1])

        radius = rotor_radius + (alpha * ydistance)

        xmin = p_j[0] - radius
        xmax = p_j[0] + radius
    else:
        print(0)
        return False

    if xmin < p_i[0] and xmax > p_i[0]:
        return True
    else:
        return False


def calc_coefficients(location_x, location_y, wind_speeds, prob_l, n, angle, n_wind):
    # Epsilon contains pairs of cells where Turbines cannot be placed due to proximity.
    # Smaller than dist x step (meters) is not allowed. Absolute distance is used.
    abs_dist_x = np.abs(location_x[:, None] - location_x)
    abs_dist_y = np.abs(location_y[:, None] - location_y)
    mask = np.triu((abs_dist_x < 200) & (abs_dist_y < 200), k=1)
    i, j = np.where(mask)
    Epsilon = list(zip(i, j))

    theta = ((np.arange(0, len(wind_speeds)) % n_wind) * angle) * math.pi / 180
    temp_x = np.outer(np.cos(theta), location_x) - np.outer(np.sin(theta), location_y)
    temp_y = np.outer(np.sin(theta), location_x) + np.outer(np.cos(theta), location_y)
    a = 0.326795
    alpha = 0.09437
    rotor_radius = 27.881
    y_dist = temp_y[:, :, None] - temp_y[:, None, :]
    radius = rotor_radius + (alpha * np.abs(y_dist))
    radius = np.transpose(radius, (0, 2, 1))
    xmin = np.transpose(temp_x[:, np.newaxis, :] - radius, (0, 2, 1))
    xmax = np.transpose(temp_x[:, np.newaxis, :] + radius, (0, 2, 1))
    wakes = ((xmin < temp_x[:, np.newaxis, :]) & (xmax > temp_x[:, np.newaxis, :])) * (
        y_dist > 0
    )
    wakes = np.transpose(wakes, (0, 2, 1))
    indices = np.array(np.where(wakes))
    U = {}

    for l in range(len(wind_speeds)):
        theta_idx = indices[:, indices[0] == l][1:]
        U.update(
            {(i, l): theta_idx[:, theta_idx[0] == i][-1].tolist() for i in range(n)}
        )

    u_inf = wind_speeds
    denom = ((alpha * np.abs(y_dist) / rotor_radius) + 1) ** 2
    velocity2 = u_inf[:, np.newaxis, np.newaxis] * (1 - (2 * a / denom)) * wakes
    energy_coef2 = (
        0.33 * (u_inf[:, np.newaxis, np.newaxis] ** 3 - velocity2**3) * wakes
    )
    ss_coef2 = ((1 - velocity2 / u_inf[:, np.newaxis, np.newaxis]) ** 2) * wakes

    aggregated_coef2 = np.sum(prob_l[:, np.newaxis, np.newaxis] * energy_coef2, axis=0)

    return aggregated_coef2, ss_coef2, Epsilon, U, velocity2


def qubo_fy(wind_speeds, prob_l, n, aggregated_coef, m, Epsilon, lam_=10500):
    # setting a penalty term

    bqm = dimod.BinaryQuadraticModel("BINARY")
    var = np.array([f"x{i}" for i in range(n)])
    bqm.add_linear_from(
        dict(zip(var, [-0.33 * np.dot(prob_l, wind_speeds**3)] * len(var)))
    )

    for i in range(n):
        for j in range(n):
            # only basic velocity componenet
            if i != j:
                bqm.add_quadratic(var[i], var[j], aggregated_coef[i, j])

    bqm.add_linear_equality_constraint(list(zip(var, [1.0] * n)), lam_, -m)

    for i in range(n):
        for j in range(n):
            # must not locate turbines too closely
            if (i, j) in Epsilon:
                bqm.add_quadratic(
                    var[i],
                    var[j],
                    lam_,
                )

    return bqm


def log_fn(state, sampler_type):
    # if sampler_type == "pasqal":
    #     print(state.subsamples.info, end="\n\n")
    #     pass
    log_dict = {"sampler_type": sampler_type}

    # log samples
    if sampler_type == "race":
        samples = [s.samples.to_serializable(pack_samples=False) for s in state]
    else:
        samples = state.samples.to_serializable(pack_samples=False)
    log_dict["samples"] = json.loads(json.dumps(samples))

    # log subsamples
    if sampler_type in ["pasqal", "composer"]:
        subsamples = state.subsamples.to_serializable(pack_samples=False)
        log_dict["subsamples"] = json.loads(json.dumps(subsamples))

    return log_dict


def SimplifiedQbsolv(
    file,
    sampler,
    max_iter=10,
    max_time=None,
    convergence=3,
    energy_threshold=None,
    max_subproblem_size=30,
):
    """Races a Tabu solver and a Neutral atoms-based sampler of flip-energy-impact induced
    subproblems.
    """

    energy_reached = None
    if energy_threshold is not None:
        energy_reached = lambda en: en <= energy_threshold

    print(str(sampler))

    tabu_log = Log(key=partial(log_fn, sampler_type="tabu"), outfile=file)
    sampler_log = Log(key=partial(log_fn, sampler_type="pasqal"), outfile=file)
    compose_log = Log(key=partial(log_fn, sampler_type="composer"), outfile=file)
    race_log = Log(key=partial(log_fn, sampler_type="race"), outfile=file)
    argmin_log = Log(key=partial(log_fn, sampler_type="argmin"), outfile=file)
    trackmin_log = Log(key=partial(log_fn, sampler_type="trackmin"), outfile=file)

    sampler_log.memo = False
    compose_log.memo = False
    race_log.memo = False
    argmin_log.memo = False
    trackmin_log.memo = False

    workflow = Loop(
        Race(
            InterruptableTabuSampler() | tabu_log,
            EnergyImpactDecomposer(
                size=max_subproblem_size,
                rolling=True,
                rolling_history=0.75,
                traversal="energy",
            )
            | sampler
            | sampler_log
            | SplatComposer()
            | compose_log,
        )
        | race_log
        | ArgMin()
        | argmin_log
        | TrackMin(output=True)
        | trackmin_log,
        max_iter=max_iter,
        max_time=max_time,
        convergence=convergence,
        terminate=energy_reached,
    )

    return workflow


def simulated_anneal(bqm, U, problem_config):
    start_t = time.time()

    print("Num variables:", bqm.num_variables)

    max_iter = 10
    max_time = None
    convergence = 3
    max_subproblem_size = 10

    seed = 42
    run_options = {"nsteps": 2000, "num_cpus": 12}
    qaa_config = {
        "duration": 50000,
        "sampling_rate": 0.1,
        "with_modulation": False,
        "run_options": run_options,
        "n_samples": 1000,
        "seed": seed,
    }

    qaoa_config = {
        "duration": 1000,
        "sampling_rate": 0.1,
        "with_modulation": False,
        "run_options": run_options,
        "n_samples": 10000,
        "n_layers": 4,
        "max_iter": 10,
        "n_repetitions": 1,
        "seed": seed,
    }

    # sampler = TabuSubproblemSampler(timeout=50 * 1000)
    # sampler = SimulatedAnnealingSubproblemSampler()
    # sampler = PasqalQAOASubproblemSampler(**qaoa_config)
    sampler = PasqalQAASubproblemSampler(**qaa_config)

    file_name = f"./quantum_results/{str(sampler)}_{problem_config['m']}_{problem_config['n_wind']}_{problem_config['axis']}.txt"
    file = open(file_name, "w")

    workflow = SimplifiedQbsolv(
        sampler=sampler,
        max_iter=max_iter,
        max_time=max_time,
        convergence=convergence,
        max_subproblem_size=max_subproblem_size,
        file=file,
    )

    init_state = State.from_sample(min_sample(bqm), bqm)
    response = workflow.run(init_state).result()

    ll = list(response.samples)[0]

    count = 0
    relevant_sol = []
    for k, v in ll.items():
        if v > 0:
            # print(k)
            count += 1
            relevant_sol.append(int(str(k).split("x")[1]))

    print(relevant_sol)
    print(len(relevant_sol))

    total_energy = 0
    Energy = {}
    for i in relevant_sol:
        for l in range(len(wind_speeds)):
            # only wake!!!
            Energy[(i, l)] = 0.33 * prob_l[l]
            temp = 0
            for j in relevant_sol:
                if j in U[(i, l)]:
                    temp += ss_coef[
                        l, i, j
                    ]  # pow(1 - velocity[i, j, l] / wind_speeds[l], 2)
            Energy[(i, l)] *= pow(wind_speeds[l] * (1 - pow(temp, 0.5)), 3)

    total_energy = sum(
        [Energy[(i, l)] for i in relevant_sol for l in range(len(wind_speeds))]
    )

    total_time = time.time() - start_t
    print("total_energy = ", total_energy)
    print("total time = " + str(total_time))

    file.write(
        json.dumps(
            {
                "total_energy": total_energy,
                "total_time": total_time,
                "solution": relevant_sol,
            }
        )
    )

    return total_energy, total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=20, choices=[20, 30, 40])
    parser.add_argument("--n_wind", type=int, default=1, choices=[1, 36])
    parser.add_argument("--axis", type=int, default=10, choices=[10, 20])

    args = parser.parse_args()

    m = args.m
    n_wind = args.n_wind
    axis_ = args.axis

    n = pow(axis_, 2)
    if axis_ == 10:
        step = 200  # cell size in meters
    elif axis_ == 20:
        step = 100
    else:
        step = 100

    angle = 360 / n_wind

    problem_config = {"m": m, "n_wind": n_wind, "axis": axis_}

    print("Calculating locations...")
    location_x, location_y = calc_location(axis_, step)

    print("Setting wind regime...")
    wind_speeds, prob_l = set_wind_regime(n_wind)

    print("Computing coefficients...")
    aggregated_coef, ss_coef, Epsilon, U, velocity = calc_coefficients(
        location_x, location_y, wind_speeds, prob_l, n, angle, n_wind
    )

    print("N wind:", n_wind)
    print("Number of turbines: " + str(m))
    bqm = qubo_fy(wind_speeds, prob_l, n, aggregated_coef, m, Epsilon)
    SA_result = simulated_anneal(bqm, U, problem_config)
