# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:31:55 2025

@author: dleon
"""

from functools import partial
import numpy as np
import networkx as nx
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
from QL import init_strats, run_ql, check_var, check_rel_diff
from networkgames import generate_edgeset, shapley, sato, conflict


def run_single_experiment(params):
    """
    Run a single experiment and check (numerical) convergence.
    """
    n_agents, n_actions, n_iter, p, T, game_type = params
    window_size = int(0.1 * n_iter)

    network = nx.erdos_renyi_graph(n_agents, p)
    edgeset = generate_edgeset(network)
    n_edges = network.number_of_edges()

    GAME_MAP = {
        "shapley": shapley,
        "sato": sato,
        "conflict": partial(conflict, edgeset=edgeset, n_agents=n_agents, n_actions=n_actions)
    }

    game = GAME_MAP[game_type](n_edges=n_edges)

    x = init_strats(n_agents, n_actions)
    Q = np.zeros((n_agents, n_actions))
    x, Q, traj = run_ql(x, Q, n_iter, edgeset, game, n_agents, n_actions, T=T)
    traj = traj.T

    return check_var(traj, window_size, 1e-2) and check_rel_diff(traj, window_size, 1e-5)


def get_initial_T(n_agents, p):
    """
    Read starting T value from a txt file. npy arrays were coorrupted, so have to resort to this.
    """
    try:
        df = pd.read_csv(f'min_temp_results_shapley_p{p}.txt', sep='[:,=]', engine='python',
                         names=['drop', 'N', 'drop2', 'T']).drop(['drop', 'drop2'], axis=1)
        T_value = df[df['N'] == n_agents]['T'].iloc[0]
        return float(T_value)
    except (FileNotFoundError, IndexError):
        raise ValueError


def find_min_temperature(n_agents, p, game_type='shapley', n_actions=3, n_iter=3000,
                         n_experiments=15, T_step=1/64, max_T=10.0):
    """
    Find minimum temperature for 100% convergence for given number of agents.
    """
    # Get initial temperature from file
    initial_T = get_initial_T(n_agents, p)
    if initial_T is None:
        raise ValueError(
            f"Could not find initial T for N={n_agents}, p={p}. Using default.")

    current_T = initial_T
    print(f"Starting search at temperature = {current_T:.6f}")

    while current_T <= max_T:
        print(current_T)
        params_list = [(n_agents, n_actions, n_iter, p, current_T, game_type)
                       for _ in range(n_experiments)]

        # Run experiments in parallel
        with mp.Pool() as pool:
            results = pool.map(run_single_experiment, params_list)

        # Check if ALL experiments converged
        if all(results):
            return current_T

        current_T += T_step

    return None


def main():
    # Parameters
    p = 0.2  # Fixed probability
    agent_range = range(15, 90, 5)
    game_type = 'shapley'
    n_actions = 3
    n_iter = 4000
    n_experiments = 48
    T_step = 1/64

    results = {}

    # Run for each number of agents
    for n_agents in tqdm(agent_range, desc="Processing different network sizes"):
        min_T = find_min_temperature(
            n_agents=n_agents,
            p=p,
            game_type=game_type,
            n_actions=n_actions,
            n_iter=n_iter,
            n_experiments=n_experiments,
            T_step=T_step
        )

        results[n_agents] = min_T
        print(f"N={n_agents}: Minimum T={min_T}")

        # Save results after each network size
        np.save(f'min_temp_results_100pct_{game_type}_p{p}.npy', results)

        # Also save as readable text file
        with open(f'min_temp_results_100pct_{game_type}_p{p}.txt', 'w') as f:
            for n, t in sorted(results.items()):
                f.write(f"N={n}: T={t}\n")


if __name__ == "__main__":
    main()
