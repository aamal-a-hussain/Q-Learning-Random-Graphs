import json
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from parameters import ExperimentParameters


def load_parameters(root_path, folder):
    with open(os.path.join(root_path, folder, "parameters.json"), "r") as f:
        params = json.load(f)

    # game_parameters = {
    #     "n_agents": params["n_agents"],
    #     "n_actions": params["n_actions"],
    #     "game_type": params["game_type"],
    #     "n_iter": params["n_iter"],
    # }
    # network_parameters = {"network_type": "sbm", "q": params["network_parameters"]["q"]}
    # params = {
    #     "game_parameters": game_parameters,
    #     "network_parameters": network_parameters,
    #     "nP": 32,
    #     "nT": 32,
    #     "n_expt": 12,
    #     "p_range": [0.1, 1],
    #     "T_range": [0.1, 4.5],
    # }
    params = ExperimentParameters(**params)

    return params


def create_heatmap_subplots(root_path):
    Qs = {"2": 0.2, "4": 0.4, "6": 0.6, "8": 0.8}
    Ns = [15, 20, 25]
    heatmap_fig = make_subplots(
        rows=len(Qs),
        cols=len(Ns),
        horizontal_spacing=0.05,
        vertical_spacing=0.03,  # Reduce vertical spacing between rows
        subplot_titles=[f"N = {N}, q = {q}" for q in Qs.values() for N in Ns],
    )

    for row, k in tqdm(enumerate(Qs.keys())):
        row += 1
        for col, N in tqdm(enumerate(Ns)):
            col += 1
            heatmap_npy = np.load(
                os.path.join(
                    root_path,
                    f"heatmap_nagents_{N}_shapley_{k}",
                    "heatmap_iteration_3.npy",
                )
            )
            params = load_parameters(root_path, f"heatmap_nagents_{N}_shapley_{k}")
            hmp = plot_heatmap(heatmap_npy, params)
            heatmap_fig.add_trace(hmp, row=row, col=col)

            heatmap_fig.update_xaxes(
                range=[params.T_range[0], params.T_range[1]],
                row=row,
                col=col,
                title_text="Exploration Rate (T)" if col == 1 and row == 4 else "",
            )
            heatmap_fig.update_yaxes(
                range=[params.p_range[0], params.p_range[1]],
                row=row,
                col=col,
                title_text="Edge Probability (p)" if col == 1 and row == 4 else "",
            )

    heatmap_fig.update_layout(
        height=800 * len(Qs),
        width=800 * len(Ns),
        showlegend=False,
        # title_text="Network Shapley Game: Stochastic Block Model",
        title_x=0.5,
        font=dict(
            family="CMU Serif Bold",
            size=32,
        ),
        margin=dict(t=60, l=60, r=30, b=60),  # Adjust margins
    )

    for annotation in heatmap_fig.layout.annotations:
        annotation.font = dict(
            size=32,
            family="CMU Serif",
            color="black",
        )

    heatmap_fig.update_traces(
        selector=dict(type="heatmap"),
        colorbar=dict(
            title="Value",
            tickfont=dict(size=32),
            len=0.2,
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


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    fig = create_heatmap_subplots(".results/results_SBM")
    fig.write_html("plots/sbm_subplots.html")
