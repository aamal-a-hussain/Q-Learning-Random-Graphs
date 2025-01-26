import os
import json
import numpy as np
import plotly.graph_objects as go

from parameters import ExperimentParameters

PATH = ".results/final_results_sato/sato_heatmap_nagents_30/"


def plot_heatmaps(heatmap, params: ExperimentParameters):
    nP, nT = heatmap.shape
    p_range = np.linspace(params.p_range[0], params.p_range[1], nP)
    T_range = np.linspace(params.T_range[0], params.T_range[1], nT)

    fig = go.Figure(
        data=go.Heatmap(
            x=T_range,
            y=p_range,
            z=heatmap,
            colorscale="inferno",
        )
    )
    fig.update_layout(
        title="Q-Learning Convergence Heatmap",
        xaxis_title="Temperature (T)",
        yaxis_title="Probability (p)",
    )

    return fig


if __name__ == "__main__":
    heatmap = np.load(os.path.join(PATH, "heatmap_iteration_4_artefact_fixed.npy"))
    os.makedirs("plots", exist_ok=True)
    with open(os.path.join(PATH, "parameters.json"), "r") as f:
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
    fig = plot_heatmaps(heatmap, params)
    fig.write_image("plots/sato_30_agents.png")
