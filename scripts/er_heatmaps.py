import json
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

import networkgames
from parameters import ExperimentParameters


def er_bound(p, N, eps=1e-6):
    s = np.sqrt(p * (1 - p))
    return (N - 1) * p + (2 * s + eps) * np.sqrt(N)


def load_parameters(root_path, folder):
    with open(os.path.join(root_path, folder, "parameters.json"), "r") as f:
        params = json.load(f)

    game_parameters = {
        "n_agents": params["n_agents"],
        "n_actions": params["n_actions"],
        "game_type": params["game_type"],
        "n_iter": params["n_iter"],
    }
    network_parameters = {"network_type": "er"}
    params = {
        "game_parameters": game_parameters,
        "network_parameters": network_parameters,
        "nP": 32,
        "nT": 32,
        "n_expt": 12,
        "p_range": [0.1, 1],
        "T_range": [0.1, 4.5],
    }
    params = ExperimentParameters(**params)

    return params


def create_heatmap_subplots(root_path):
    subdir = os.listdir(root_path)
    subdir.sort(key=lambda x: int(x.split("_")[-1]))

    # Create subplots with more space between them and better sizing
    heatmap_fig = make_subplots(
        rows=1,
        cols=len(subdir),
        horizontal_spacing=0.05,  # Adjust space between subplots
        subplot_titles=[
            f"N = {folder.split('_')[-1]}" for folder in subdir
        ],  # Add titles
    )

    for col, folder in tqdm(enumerate(subdir)):
        heatmap_npy = np.load(
            os.path.join(root_path, folder, "heatmap_iteration_3.npy")
        )
        params = load_parameters(root_path, folder)
        # hmp, bound, params = plot_heatmap_and_bound(heatmap_npy, params)
        hmp = plot_heatmap(heatmap_npy, params)
        heatmap_fig.add_trace(hmp, row=1, col=col + 1)
        # heatmap_fig.add_trace(bound, row=1, col=col + 1)

        heatmap_fig.update_xaxes(
            range=[params.T_range[0], params.T_range[1]],
            row=1,
            col=col + 1,
            title_text="Exploration Rate (T)" if col == 0 else "",
        )
        heatmap_fig.update_yaxes(
            range=[params.p_range[0], params.p_range[1]],
            row=1,
            col=col + 1,
            title_text="Edge Probability (p)" if col == 0 else "",
        )

    heatmap_fig.update_layout(
        height=800,
        width=800 * len(subdir),
        showlegend=False,
        # title_text=f"Network {params.game_parameters.game_type.capitalize()} Game",
        title_text="Conflict Network Game",
        title_x=0.5,
        font=dict(
            family="CMU Serif Bold",
            size=32,
        ),
        margin=dict(t=60, l=60, r=30, b=60),  # Adjust margins
    )

    for i, _ in enumerate(subdir):
        heatmap_fig.layout.annotations[i].font = dict(
            size=32,
            family="CMU Serif Bold",
        )

    heatmap_fig.update_traces(
        selector=dict(type="heatmap"),
        colorbar=dict(
            title="Value",
            tickfont=dict(size=32),
            len=1,
        ),
        showscale=True,
    )

    return heatmap_fig


def plot_heatmap(heatmap, params: ExperimentParameters):
    nP, nT = heatmap.shape
    p_range = np.linspace(params.p_range[0], params.p_range[1], nP)
    T_range = np.linspace(params.T_range[0], params.T_range[1], nT)

    hmp = go.Heatmap(x=T_range, y=p_range, z=heatmap, colorscale="cividis")
    return hmp


def plot_heatmap_and_bound(heatmap, params, upper_bound=6.5):
    nP, nT = heatmap.shape
    p_range = np.linspace(params.p_range[0], params.p_range[1], nP)
    iii = networkgames.compute_iii(
        getattr(networkgames, params.game_parameters.game_type)
    )
    bound_values = [iii * er_bound(p, params.game_parameters.n_agents) for p in p_range]

    additional_T = int(
        (upper_bound - params.T_range[1]) * nT / (params.T_range[1] - params.T_range[0])
    )
    heatmap = np.hstack((heatmap, np.zeros((nP, additional_T))))
    params.T_range = [params.T_range[0], upper_bound]
    params.nP, params.nT = heatmap.shape

    hmp = plot_heatmap(heatmap, params)
    bound = go.Scatter(
        x=bound_values,
        y=p_range,
        mode="lines",
        line=dict(
            color="white",
            width=2,
            dash="dash",
        ),
        name="Boundary",
    )

    return hmp, bound, params


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    fig = create_heatmap_subplots(".results/final_results_conflict")
    fig.write_html("plots/conflict_subplots.html")
