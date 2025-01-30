import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

import QL
from networkgames import generate_edgeset, sato

N_AGENTS = 50
N_ACTIONS = 3
GAME = sato
Q = 0.1
N_BLOCKS = 2
N_ITER = 3000


def create_sbm_game(p_min, p_max):
    community_size = N_AGENTS // N_BLOCKS
    Ps = np.linspace(p_min, p_max, num=N_BLOCKS)
    sizes = [community_size] * N_BLOCKS
    block_probability_matrix = (
        Q * np.ones((N_BLOCKS, N_BLOCKS)) - Q * np.eye(N_BLOCKS) + np.diag(Ps)
    )

    block_probabilities = [list(p) for p in block_probability_matrix]

    return nx.stochastic_block_model(sizes, block_probabilities)


def run_single_game(T):
    window_size = N_ITER // 10

    network = create_sbm_game(0.2, 0.8)
    edgeset = generate_edgeset(network)
    n_edges = network.number_of_edges()

    game = GAME(n_edges=n_edges)
    x = QL.init_strats(N_AGENTS, N_ACTIONS)
    Q_values = np.zeros((N_AGENTS, N_ACTIONS))
    x, Q_values, traj = QL.run_ql(
        x, Q_values, N_ITER, edgeset, game, N_AGENTS, N_ACTIONS, T=T
    )
    traj = traj.T

    window_traj = window_traj = traj[:, -window_size:].reshape(
        (N_AGENTS, N_ACTIONS, window_size)
    )
    maximum_extents = (window_traj.max(axis=-1) - window_traj.min(axis=-1)).max(axis=-1)
    return [
        (i, e)
        for i, p in enumerate(network.graph["partition"])
        for e in maximum_extents[list(p)]
    ]


def run_multiple_experiments(T, n_expt=32):
    num_processes = mp.cpu_count()
    print(f"Processing using {num_processes} processes...")
    maximum_extents = {f"Partition {i}": [] for i in range(N_BLOCKS)}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(run_single_game, T) for _ in range(n_expt)]
        results = list(tqdm([future.result() for future in futures], total=n_expt))

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


if __name__ == "__main__":
    os.makedirs(f"histogram_{N_AGENTS}_agents", exist_ok=True)
    for T in [0.75, 1.0, 1.25, 1.5]:
        maximum_extents = run_multiple_experiments(T)
        fig = plot_histogram(maximum_extents)
        fig.write_html(f"histogram_{N_AGENTS}_agents/{T}.html")
