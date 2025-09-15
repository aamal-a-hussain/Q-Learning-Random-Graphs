import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

import QL
from parameters import ExperimentParameters
from game import SatoGame, ShapleyGame

GAME_REGISTRY = {
    "shapley": ShapleyGame,
    "sato": SatoGame,
}


def create_sbm_game(n_agents, params):
    community_size = n_agents // params.n_blocks
    Ps = np.linspace(params.p_min, params.p_max, num=params.n_blocks)
    sizes = [community_size] * params.n_blocks
    block_probability_matrix = (
        params.q * np.ones((params.n_blocks, params.n_blocks))
        - params.q * np.eye(params.n_blocks)
        + np.diag(Ps)
    )

    block_probabilities = [list(p) for p in block_probability_matrix]

    return nx.stochastic_block_model(sizes, block_probabilities)


def run_single_game(params, T):
    window_size = params.game_parameters.n_iter // 10

    network = create_sbm_game(params.game_parameters.n_agents, params.network_parameters)
    edgeset = list(network.edges())

    game = GAME_REGISTRY[params.game_parameters.game_type](params.game_parameters.n_agents)
    x = QL.init_strats(
        params.game_parameters.n_agents, params.game_parameters.n_actions
    )
    Q_values = np.zeros(
        (params.game_parameters.n_agents, params.game_parameters.n_actions)
    )
    x, Q_values, traj = QL.run_ql(
        x,
        Q_values,
        params.game_parameters.n_iter,
        edgeset,
        game,
        T=T,
    )
    traj = traj.T

    window_traj = window_traj = traj[:, -window_size:].reshape(
        (params.game_parameters.n_agents, params.game_parameters.n_actions, window_size)
    )
    maximum_extents = (window_traj.max(axis=-1) - window_traj.min(axis=-1)).max(axis=-1)
    return [
        (i, e)
        for i, p in enumerate(network.graph["partition"])
        for e in maximum_extents[list(p)]
    ]


def run_multiple_experiments(params, T):
    num_processes = mp.cpu_count()
    print(f"Processing using {num_processes} processes...")
    maximum_extents = {f"Partition {i}": [] for i in range(params.network_parameters.n_blocks)}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(run_single_game, params, T) for _ in range(params.n_expt)
        ]
        results = list(
            tqdm([future.result() for future in futures], total=params.n_expt)
        )

    for res in results:
        for r in res:
            maximum_extents[f"Partition {r[0]}"].append(r[1])

    return maximum_extents


def plot_histogram(maximum_extents):
    fig = go.Figure()
    for partition, extents in maximum_extents.items():
        fig.add_trace(
            go.Histogram(x=extents, histnorm="probability", nbinsx=100, name=partition)
        )
    fig.update_layout(
        title="Convergence in SBM Communities",
        xaxis_title="Maximum Extent",
        yaxis_title="Proportion",
        xaxis_range=[0, 1],
    )

    return fig


def main():
    params = {
        "game_parameters": {
            "n_agents": 100,
            "n_actions": 3,
            "n_iter": 2500,
            "game_type": "sato",
        },
        "network_parameters": {
            "network_type": "sbm",
            "n_blocks": 5,
            "q": 0.1,
            "p_min": 0.2,
            "p_max": 0.8,
        },
        "T_range": (0.75, 1.5),
        "nT": 4,
        "n_expt": 32
    }

    params = ExperimentParameters(**params)

    os.makedirs(f"histogram_{params.game_parameters.n_agents}_agents", exist_ok=True)
    for T in params.Ts:
        maximum_extents = run_multiple_experiments(params, T)
        fig = plot_histogram(maximum_extents)
        fig.write_html(f"histogram_{params.game_parameters.n_agents}_agents/{T}.html")


if __name__ == "__main__":
    main()