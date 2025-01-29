from functools import partial
import numpy as np
import networkx as nx
from tqdm import tqdm
import multiprocessing as mp
from QL import init_strats, run_ql, check_var, check_rel_diff
from networkgames import generate_edgeset, shapley, sato, conflict


def run_single_experiment(params):
    """
    Run a single experiment and return whether it converged.
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


def find_min_temperature(n_agents, p, last_T=None, game_type='shapley', n_actions=3, n_iter=3000,
                         n_experiments=15, initial_T=12/64, T_step=1/64, max_T=10.0):
    """
    Find minimum temperature for convergence for given number of agents.

    Parameters:
    -----------
    last_T : float or None
        The minimum temperature found for the previous network size
    """
    # Determine starting temperature
    if last_T is not None:
        # Start from (last_T - 5 temperature steps) or initial_T, whichever is larger
        current_T = max(last_T - (3 * T_step), initial_T)
    else:
        current_T = initial_T

    print(f"Starting search at temperature = {current_T:.6f}")

    while current_T <= max_T:
        # Prepare parameters for parallel experiments
        print(current_T)
        params_list = [(n_agents, n_actions, n_iter, p, current_T, game_type)
                       for _ in range(n_experiments)]

        # Run experiments in parallel
        with mp.Pool() as pool:
            results = pool.map(run_single_experiment, params_list)

        # Check if any experiment converged
        if any(results):
            return current_T

        current_T += T_step

    return None  # Return None if no convergence found up to max_T


def main():
    # Parameters
    p = 0.2  # Fixed probability
    agent_range = range(15, 105, 5)  # 15 to 100 inclusive, step 5
    game_type = 'shapley'
    n_actions = 3
    n_iter = 3000
    n_experiments = 48
    initial_T = 12/64
    T_step = 1/64

    results = {}
    last_T = None

    # Run for each number of agents
    for n_agents in tqdm(agent_range, desc="Processing different network sizes"):
        min_T = find_min_temperature(
            n_agents=n_agents,
            p=p,
            last_T=last_T,
            game_type=game_type,
            n_actions=n_actions,
            n_iter=n_iter,
            n_experiments=n_experiments,
            initial_T=initial_T,
            T_step=T_step
        )

        results[n_agents] = min_T
        last_T = min_T  # Store this T value for the next iteration

        print(f"N={n_agents}: Minimum T={min_T}")

        # Save results after each network size
        np.save(f'min_temp_results_{game_type}_p{p}.npy', results)

        # Also save as readable text file
        with open(f'min_temp_results_{game_type}_p{p}.txt', 'w') as f:
            for n, t in sorted(results.items()):
                f.write(f"N={n}: T={t}\n")


if __name__ == "__main__":
    main()
