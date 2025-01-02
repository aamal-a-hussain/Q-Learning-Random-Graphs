# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:56:51 2024

@author: dleon
"""

import plotly.graph_objects as go
import numpy as np
import networkx as nx
from tqdm import tqdm
import QL
from networkgames import generate_edgeset, shapley
import multiprocessing as mp


def run_single_cell_experiment(params):
    """
    Run all experiments for a single cell and return the convergence rate.
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

        if QL.check_var(traj, window_size, 1e-2) and QL.check_rel_diff(traj, window_size, 1e-5):
            converged += 1

    return 1 - converged / n_expt


def identify_cells_of_interest(heatmap):
    """
    Identifies cells where both convergence and divergence occurred
    (i.e., 0 < probability < 1) and their horizontal neighbors.
    Also includes all cells between any two selected cells in the same row.
    Returns a boolean mask of same shape as input heatmap.
    """
    nP, nT = heatmap.shape
    mask = np.zeros_like(heatmap, dtype=bool)

    # Find cells where both convergence and divergence occurred
    interesting_cells = (heatmap > 0) & (heatmap < 1)
    mask |= interesting_cells

    # Add horizontal neighbors (handling boundaries)
    for i in range(nP):
        for j in range(nT):
            if interesting_cells[i, j]:
                # Add left neighbor if not at left boundary
                if j > 0:
                    mask[i, j-1] = True
                # Add right neighbor if not at right boundary
                if j < nT-1:
                    mask[i, j+1] = True

    # Fill in gaps in each row
    for i in range(nP):
        # Find leftmost and rightmost True in this row
        true_positions = np.where(mask[i])[0]
        if len(true_positions) >= 2:
            # Fill everything between min and max position
            mask[i, true_positions[0]:true_positions[-1]+1] = True

    # Mark cells in the row above any marked cells
    # Create a copy of the mask to avoid affecting the iteration
    above_mask = mask.copy()
    for i in range(nP-1, 0, -1):  # Start from bottom row, move up
        for j in range(nT):
            if mask[i, j]:  # If cell is marked
                above_mask[i-1, j] = True  # Mark cell above it

    return above_mask


# def identify_cells_of_interest(heatmap):
#    """
#    Find cells where for which at least one experiment didn't converge and their horizontal neighbors.
#    We pay special attention for the first and last cell on each row, as they doný have immediate horizontal neighbors.
#    Returns a boolean mask of same shape as input heatmap.
#    """
#    nP, nT = heatmap.shape
#    mask = np.zeros_like(heatmap, dtype=bool)
#
#    # Find cells where at least one experiment didn't converge
#    non_convergent = heatmap > 0
#    mask |= non_convergent  # in place O R
#
#    # Add immediate horizontal neighbors, i.e. one to the left and one to the right
#    for i in range(nP):
#        for j in range(nT):
#            if non_convergent[i, j]:
#                # Add left neighbor if not at left boundary
#                if j > 0:
#                    mask[i, j-1] = True
#                # Add right neighbor if not at right boundary
#                if j < nT-1:
#                    mask[i, j+1] = True
#
#    # THIS IS REALLY IMPORTANT, otherwise it could happen quite often that
#    # for a cell with probability of convergence p < 1, all 10 experiments converge
#    # and we never touch it again. actually, this gives us a change to explore it again
#    # if we expand the grid enough times
#    # Fill in gaps in each row
#    for i in range(nP):
#        # Find leftmost and rightmost True in this row
#        true_positions = np.where(mask[i])[0]
#        if len(true_positions) >= 2:
#            # Fill everything between min and max position
#            mask[i, true_positions[0]:true_positions[-1]+1] = True
#
#    return mask


def create_refined_heatmap(original_heatmap, mask, n_agents, n_actions, nP, nT,
                           n_iter, n_expt, p_range, T_range, game_type):
    """
    Creates a new heatmap with doubled nP and doubled nT.

    Parameters:
    -----------
    original_heatmap : numpy.ndarray
        The original heatmap to refine
    mask : numpy.ndarray
        Boolean mask indicating which cells to recalculate
    n_agents : int
        Number of agents in the network
    n_actions : int
        Number of actions available to each agent
    nP : int
        Number of p values in original heatmap
    nT : int
        Number of T values in original heatmap
    n_iter : int
        Number of iterations for Q-Learning
    n_expt : int
        Number of experiments per cell
    p_range : tuple
        (min_p, max_p) for probability range
    T_range : tuple
        (min_T, max_T) for temperature range
    game_type : str
        Type of game ('shapley' or 'sato')

    Returns:
    --------
    numpy.ndarray
        The refined heatmap with doubled resolution
    """
    new_nP = 2 * nP
    new_nT = 2 * nT

    # Create new parameter grids with doubled resolution
    new_ps = np.linspace(p_range[0], p_range[1], new_nP)
    new_Ts = np.linspace(T_range[0], T_range[1], new_nT)

    # Initialize new heatmap
    refined_heatmap = np.zeros((new_nP, new_nT))

    # Create list of cells that need experiments
    cells_to_process = []
    for i in range(nP):
        for j in range(nT):
            if mask[i, j]:
                new_i1, new_i2 = 2*i, 2*i + 1
                new_j1, new_j2 = 2*j, 2*j + 1
                for new_i in [new_i1, new_i2]:
                    for new_j in [new_j1, new_j2]:
                        cells_to_process.append((
                            new_ps[new_i],
                            new_Ts[new_j],
                            n_agents,
                            n_actions,
                            n_iter,
                            n_expt
                        ))

    # Process cells in parallel using a single pool
    print(f'Processing {len(cells_to_process)} cells in parallel...')
    num_processes = mp.cpu_count()
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(run_single_cell_experiment, cells_to_process),
            total=len(cells_to_process)
        ))

    # Fill in the results
    result_idx = 0
    for i in range(nP):
        for j in range(nT):
            new_i1, new_i2 = 2*i, 2*i + 1
            new_j1, new_j2 = 2*j, 2*j + 1
            if mask[i, j]:
                for new_i in [new_i1, new_i2]:
                    for new_j in [new_j1, new_j2]:
                        refined_heatmap[new_i, new_j] = results[result_idx]
                        result_idx += 1
            else:
                # Copy the original value to all four subcells
                refined_heatmap[new_i1:new_i2+1,
                                new_j1:new_j2+1] = original_heatmap[i, j]

    return refined_heatmap


def main():
    # Define parameters
    n_agents = 15
    n_actions = 3
    nP = 20
    nT = 20
    n_iter = 2500
    n_expt = 10
    p_range = (0.01, 1)
    T_range = (0.01, 3.5)
    game_type = 'shapley'

    # Load the original heatmap
    original_heatmap = np.load('QL-erdos-renyi-refined-2d.npy')

    # Identify cells of interest
    mask = identify_cells_of_interest(original_heatmap)

    # Print number of cells to process
    cells_to_process = np.sum(mask)
    print(f'Found {cells_to_process} cells of interest (including neighbors)')

    # Create refined heatmap with new parameters
    refined_heatmap = create_refined_heatmap(
        original_heatmap,
        mask,
        n_agents=n_agents,
        n_actions=n_actions,
        nP=nP,
        nT=nT,
        n_iter=n_iter,
        n_expt=n_expt,
        p_range=p_range,
        T_range=T_range,
        game_type=game_type
    )

    # Save the refined heatmap
    np.save('QL_erdos_renyi_refined.npy', refined_heatmap)

    # Plot the results using plotly
    nP, nT = original_heatmap.shape
    new_Ts = np.linspace(T_range[0], T_range[1], 2*nT)
    new_ps = np.linspace(p_range[0], p_range[1], 2*nP)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=new_Ts, y=new_ps, z=refined_heatmap))

    fig.write_html("QL_erdos_renyi_figure_refined.html")


if __name__ == "__main__":
    main()
