import numpy as np
import networkx as nx
from tqdm import tqdm
import plotly.graph_objects as go
import QL
from networkgames import (
    generate_edgeset,
    shapley,
    sato,
    mismatching,
    chakraborty,
    compute_iii,
)
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

N_AGENTS = 30
N_ACTIONS = 3
GAME = sato
N_ITER = 25000


def run_single_game():
    window_size = 300

    network = nx.erdos_renyi_graph(N_AGENTS, 0.8)
    edgeset = generate_edgeset(network)
    n_edges = network.number_of_edges()

    game = GAME(n_edges=n_edges)
    x = QL.init_strats(N_AGENTS, N_ACTIONS)
    Q_values = np.zeros((N_AGENTS, N_ACTIONS))
    G = nx.to_numpy_array(network)
    T = compute_iii(GAME) * np.linalg.matrix_norm(G, ord=2) + 0.05
    x, Q_values, traj = QL.run_ql(
        x, Q_values, N_ITER, edgeset, game, N_AGENTS, N_ACTIONS, T=T, verbose=True
    )
    traj = traj.T

    window_traj = window_traj = traj[:, -window_size:].reshape(
        (N_AGENTS, N_ACTIONS, window_size)
    )
    maximum_extents = (window_traj.max(axis=-1) - window_traj.min(axis=-1)).max(axis=-1)
    return [(i, e) for i, e in enumerate(maximum_extents)]


def run_multiple_experiments(n_expt=1):
    num_processes = mp.cpu_count()
    print(f"Processing using {num_processes} processes...")
    maximum_extents = {f"Agent {i}": [] for i in range(N_AGENTS)}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(run_single_game) for _ in range(n_expt)]
        results = list(tqdm([future.result() for future in futures], total=n_expt))

    for res in results:
        for r in res:
            maximum_extents[f"Agent {r[0]}"].append(r[1])

    return maximum_extents


def plot_histogram(maximum_extents):
    fig = go.Figure()
    for partition, extents in maximum_extents.items():
        fig.add_trace(go.Histogram(x=extents, histnorm="probability", name=partition))
    fig.update_layout(
        title="Convergence in SBM Communities",
        xaxis_title="Maximum Extent",
        yaxis_title="Proportion",
        xaxis_range=[0, 1],
    )

    return fig


if __name__ == "__main__":
    os.makedirs(f"er_histogram_{N_AGENTS}_agents", exist_ok=True)
    maximum_extents = run_single_game()
    fig = plot_histogram(maximum_extents)
    fig.write_html(f"er_histogram_{N_AGENTS}_agents/{T}.html")
