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
from networkgames import generate_edgeset
from game import ConflictGame, SatoGame, ShapleyGame


def run_single_experiment(params):
    """
    Run a single experiment and check (numerical) convergence.
    """
    n_agents, n_actions, n_iter, p, T, game_type = params
    window_size = int(0.1 * n_iter)
    
    # Generate network
    network = nx.erdos_renyi_graph(n_agents, p)
    edgeset = generate_edgeset(network)
    
    # Create game using the new class-based approach
    GAME_REGISTRY = {
        "shapley": ShapleyGame,
        "sato": SatoGame,
        "conflict": partial(ConflictGame, n_actions=n_actions, edgeset=edgeset),
    }
    game = GAME_REGISTRY[game_type](n_agents=n_agents)
    
    # Initialize and run Q-learning
    x = init_strats(n_agents, n_actions)
    Q = np.zeros((n_agents, n_actions))
    x, Q, traj = run_ql(x, Q, n_iter, edgeset, game, T=T)
    traj = traj.T
    
    # Check convergence
    return check_var(traj, window_size, 1e-2) and check_rel_diff(traj, window_size, 1e-5)


def get_initial_T(game_type, n_agents, p, step_size=5):
    """
    Read starting T value from the previous N value, or 1/64 if no previous value exists.
    """
    previous_n = n_agents - step_size
    
    try:
        df = pd.read_csv(f'min_temp_results_100pct_{game_type}_p{p}.txt', sep='[:,=]', engine='python',
                         names=['drop', 'N', 'drop2', 'T']).drop(['drop', 'drop2'], axis=1)
        T_value = df[df['N'] == previous_n]['T'].iloc[0]
        print(f"Starting from previous N={previous_n} T value: {T_value}")
        return float(T_value)
    except (FileNotFoundError, IndexError):
        print(f"No previous T value found for N={previous_n}, starting at T = 1/64")
        return 1/64

def find_min_temperature(n_agents, p, game_type='shapley', n_actions=3, n_iter=3000,
                         n_experiments=15, T_step=1/64, max_T=10.0):
    """
    Find minimum temperature for 100% convergence for given number of agents.
    """
    # Get initial temperature from file
    initial_T = get_initial_T(game_type, n_agents, p)
    current_T = initial_T
    print(f"Starting search at temperature = {current_T:.6f}")
    
    while current_T <= max_T:
        print(f"Testing T = {current_T:.6f}")
        
        # Create parameter list for all experiments
        params_list = [(n_agents, n_actions, n_iter, p, current_T, game_type)
                       for _ in range(n_experiments)]
        
        # Run experiments in parallel
        with mp.Pool() as pool:
            results = pool.map(run_single_experiment, params_list)
        
        # Check if ALL experiments converged
        convergence_rate = sum(results) / len(results)
        print(f"Convergence rate: {convergence_rate:.3f} ({sum(results)}/{len(results)})")
        
        if all(results):
            print(f"Found minimum temperature: {current_T:.6f}")
            return current_T
        
        current_T += T_step
    
    print(f"No temperature found up to max_T = {max_T}")
    return None


def main():
    # Parameters
    p = 0.2  # Fixed probability
    agent_range = range(25, 200, 5)
    game_type = 'shapley'
    n_actions = 3
    n_iter = 4000
    n_experiments = 50
    T_step = 1/64
    
    results = {}
    
    # Run for each number of agents
    for n_agents in tqdm(agent_range, desc="Processing different network sizes"):
        print(f"\n--- Processing N = {n_agents} agents ---")
        
        try:
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
            
        except ValueError as e:
            print(f"Error processing N={n_agents}: {e}")
            results[n_agents] = None
            continue
        
        # Save results after each network size
        np.save(f'min_temp_results_100pct_{game_type}_p{p}.npy', results)
        
        # Also save as readable text file
        with open(f'min_temp_results_100pct_{game_type}_p{p}.txt', 'w') as f:
            for n, t in sorted(results.items()):
                if t is not None:
                    f.write(f"N={n}: T={t}\n")
                else:
                    f.write(f"N={n}: T=None (failed)\n")


if __name__ == "__main__":
    main()