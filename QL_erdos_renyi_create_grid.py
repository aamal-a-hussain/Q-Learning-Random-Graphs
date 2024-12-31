# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:21:33 2024

@author: dleon
"""

import numpy as np
import networkx as nx
from tqdm import tqdm
import multiprocessing as mp
import plotly.graph_objects as go
import QL
from networkgames import generate_edgeset, shapley, sato


def process_single_cell(params):
    """
    Process a single cell with given parameters.
    Returns the non-convergence rate for the cell.
    """
    p, T, n_agents, n_actions, n_iter, n_expt = params
    window_size = int(0.1 * n_iter)
    converged = 0

    for _ in range(n_expt):
        network = nx.erdos_renyi_graph(n_agents, p)
        edgeset = generate_edgeset(network)
        n_edges = network.number_of_edges()
        game = shapley(n_edges=n_edges)

        x = QL.init_strats(n_agents, n_actions)
        Q = np.zeros((n_agents, n_actions))
        x, Q, traj = QL.run_ql(x, Q, n_iter, edgeset,
                               game, n_agents, n_actions, T=T)
        traj = traj.T

        converged += QL.check_var(traj, window_size,
                                  1e-2) and QL.check_rel_diff(traj, window_size, 1e-5)

    return 1 - converged / n_expt


def generate_heatmap(n_agents=15, n_actions=3, nP=20, nT=20, n_iter=2500, n_expt=10,
                     p_range=(0.1, 1), T_range=(0.1, 3.5), game_type='shapley',
                     save_path=None):
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

    assert game_type == 'shapley', 'sato not yet implemented, please make sure to amend code everywhere shapley appears'
    # Initialize parameters
    ps = np.linspace(p_range[0], p_range[1], nP)
    Ts = np.linspace(T_range[0], T_range[1], nT)
    hmp = np.zeros((nP, nT))

    # Create parameter list for all cells
    params_list = []
    for i, p in enumerate(ps):
        for j, T in enumerate(Ts):
            params_list.append((p, T, n_agents, n_actions, n_iter, n_expt))

    # Set up multiprocessing
    num_processes = mp.cpu_count()


    # Process cells in parallel
    print(
        f"Processing {len(params_list)} cells using {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_cell, params_list),
            total=len(params_list)
        ))

    # Reshape results into heatmap
    for idx, result in enumerate(results):
        i = idx // nT
        j = idx % nT
        hmp[i, j] = result

    # Save results if path provided
    if save_path:
        np.save(save_path, hmp)
        print(f"Heatmap saved to {save_path}")

    return hmp


def plot_heatmap(heatmap, p_range=(0.1, 1), T_range=(0.1, 3.5)):
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
    Ts = np.linspace(T_range[0], T_range[1], nT)
    ps = np.linspace(p_range[0], p_range[1], nP)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=Ts, y=ps, z=heatmap))
    fig.update_layout(
        title="Q-Learning Convergence Heatmap",
        xaxis_title="Temperature (T)",
        yaxis_title="Probability (p)"
    )
    return fig


def main():
    # Example usage
    heatmap = generate_heatmap(
        n_agents=15,
        n_actions=3,
        nP=20,
        nT=20,
        n_iter=2500,
        n_expt=10,
        save_path='QL-erdos-renyi.npy'
    )

    # Plot the results
    fig = plot_heatmap(heatmap)
    fig.show()


if __name__ == "__main__":
    main()
