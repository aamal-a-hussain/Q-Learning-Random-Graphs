# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:21:33 2024

@author: dleon
"""

import multiprocessing as mp
from functools import partial

import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

import QL
from game import ConflictGame, SatoGame, ShapleyGame
from networkgames import (
    generate_edgeset,
    generate_er_network,
    generate_sbm_network,
)
from parameters import ExperimentParameters, RunParameters


def process_single_cell(params: RunParameters):
    """
    Process a single cell with given parameters.
    Returns the non-convergence rate for the cell.
    """

    network_generator = (
        generate_sbm_network
        if params.network_parameters.network_type == "sbm"
        else generate_er_network
    )
    converged = 0
    window_size = params.n_iter // 10

    for _ in range(params.n_expt):
        network = network_generator(params.n_agents, params.network_parameters)
        edgeset = generate_edgeset(network)

        GAME_REGISTRY = {
            "shapley": ShapleyGame,
            "sato": SatoGame,
            "conflict": partial(
                ConflictGame, n_actions=params.n_actions, edgeset=edgeset
            ),
        }
        game = GAME_REGISTRY[params.game_type](n_agents=params.n_agents)

        x = QL.init_strats(params.n_agents, params.n_actions)
        Q = np.zeros((params.n_agents, params.n_actions))
        x, Q, traj = QL.run_ql(
            x,
            Q,
            params.n_iter,
            edgeset,
            game,
            T=params.T,
        )
        traj = traj.T

        converged += QL.check_var(traj, window_size, 1e-2) and QL.check_rel_diff(
            traj, window_size, 1e-5
        )

    return 1 - converged / params.n_expt


def generate_heatmap(params: ExperimentParameters, save_path=None):
    """
    Generate a heatmap of Q-Learning convergence rates.

    Parameters:
    -----------
    n_agents : int
        Number of agents in the network
    n_actions : int
        Number of actions available to each agent
    nP : int
        Number of p values (rows in heatmap)
    nT : int
        Number of T values (columns in heatmap)
    n_iter : int
        Number of iterations for Q-Learning
    n_expt : int
        Number of experiments per cell
    p_range : tuple
        (min_p, max_p) for probability range
    T_range : tuple
        (min_T, max_T) for temperature range
    game_type : str
        'shapley' or 'sato'
    save_path : str or None
        Path to save the numpy array. If None, doesn't save

    Returns:
    --------
    numpy.ndarray
        The generated heatmap
    """

    # Initialize parameters
    ps = np.linspace(params.p_range[0], params.p_range[1], params.nP)
    Ts = np.linspace(params.T_range[0], params.T_range[1], params.nT)
    hmp = np.zeros((params.nP, params.nT))

    # Create parameter list for all cells
    params_list = []
    for i, p in enumerate(ps):
        for j, T in enumerate(Ts):
            run_params = RunParameters(
                T=T,
                n_agents=params.game_parameters.n_agents,
                n_actions=params.game_parameters.n_actions,
                n_iter=params.game_parameters.n_iter,
                game_type=params.game_parameters.game_type,
                n_expt=params.n_expt,
                network_parameters=params.network_parameters.model_copy(
                    update={"p": p}
                ),
            )
            params_list.append(run_params)

    # Set up multiprocessing
    num_processes = mp.cpu_count()

    # Process cells in parallel
    print(f"Processing {len(params_list)} cells using {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(pool.imap(process_single_cell, params_list), total=len(params_list))
        )

    # Reshape results into heatmap
    for idx, result in enumerate(results):
        i = idx // params.nT
        j = idx % params.nT
        hmp[i, j] = result

    # Save results if path provided
    if save_path:
        np.save(save_path, hmp)
        print(f"Heatmap saved to {save_path}")

    return hmp


def plot_heatmap(heatmap, params: ExperimentParameters):
    """
    Plot a heatmap using plotly.

    Parameters:
    -----------
    heatmap : numpy.ndarray
        The heatmap to plot
    p_range : tuple
        (min_p, max_p) for probability range
    T_range : tuple
        (min_T, max_T) for temperature range
    """
    nP, nT = heatmap.shape
    Ts = np.linspace(params.T_range[0], params.T_range[1], nT)
    ps = np.linspace(params.p_range[0], params.p_range[1], nP)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=Ts, y=ps, z=heatmap))
    fig.update_layout(
        title="Q-Learning Convergence Heatmap",
        xaxis_title="Temperature (T)",
        yaxis_title="Probability (p)",
    )
    return fig


def main():
    # Example usage

    params = {
        "game_parameters": {
            "n_agents": 15,
            "n_actions": 3,
            "n_iter": 2500,
            "game_type": "shapley",
        },
        "network_parameters": {"network_type": "er"},
        "nP": 20,
        "nT": 20,
        "n_expt": 10,
    }

    params = ExperimentParameters(**params)
    heatmap = generate_heatmap(
        params,
        save_path="QL-erdos-renyi.npy",
    )

    # Plot the results
    fig = plot_heatmap(heatmap, params)
    fig.show()


if __name__ == "__main__":
    main()
