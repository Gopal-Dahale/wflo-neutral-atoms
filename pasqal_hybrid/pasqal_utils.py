# pylint: disable-all

"""Pasqal Sampler utilities"""

import numpy as np
from pulser.devices import Chadoq2
from scipy.optimize import minimize
from scipy.sparse import coo_array
from scipy.spatial.distance import pdist, squareform


def get_qubo_matrix(h, coupler_starts, coupler_ends, coupler_weights):
    # Find indices where array2 elements are less than array1 elements
    swap_indices = np.where(coupler_ends < coupler_starts)

    # Swap elements between the two arrays
    coupler_starts[swap_indices], coupler_ends[swap_indices] = (
        coupler_ends[swap_indices],
        coupler_starts[swap_indices],
    )

    num_vars = len(h)
    Q = coo_array(
        (coupler_weights, (coupler_starts, coupler_ends)),
        shape=(num_vars, num_vars),
    ).toarray()  # upper triangular matrix
    Q += np.diag(h)

    # print(np.round(Q, 2))

    max_val = np.max(np.abs(Q))

    # scale qubo matrix to fit on chadoq2 device
    # the maximum amplitude on chadoq2 device is 15.707963267948966
    # Max Absolute Detuning: 125.66370614359172

    max_amp = 15.7
    scale_factor = max_amp / max_val
    Q *= scale_factor

    detuning_factor = np.max(np.abs(np.diag(Q)))
    return Q, scale_factor, detuning_factor


def evaluate_mapping(new_coords, *args):
    Q, shape = args
    new_coords = np.reshape(new_coords, shape)
    pdists = pdist(new_coords)
    new_Q = np.triu(squareform(Chadoq2.interaction_coeff / pdists**6))
    # try except
    # Add a continuos penalty terms for coordinates that are too close/far together
    penalty = np.sum(1e5 * (np.exp(4 - pdists) + np.exp(pdists - 50)))
    # penalty1 = np.sum(np.where(pdists < 4, 1e5, 0))
    # penalty2 = np.sum(np.where(pdists > 50, 1e5, 0))
    res = np.linalg.norm(new_Q - Q) + penalty
    # print(res, penalty)
    return res


def embed_qubo(qubo_mat, seed):
    # Embedding a QUBO onto an atomic register
    num_vars = len(qubo_mat)
    shape = (num_vars, 2)
    # np.random.seed(seed)
    x0 = np.random.random(shape).flatten()

    res = minimize(
        evaluate_mapping,
        x0,
        args=(qubo_mat, shape),
        method="Nelder-Mead",
        tol=1e-6,
        options={"maxiter": 200000, "maxfev": None},
    )

    coords = np.reshape(res.x, shape)
    return coords
