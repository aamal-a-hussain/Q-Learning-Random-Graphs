import numpy as np
import plotly.graph_objects as go
import networkgames
from QL_erdos_renyi_create_grid import plot_heatmap
from parameters import ExperimentParameters


def er_bound(p, N, eps=1e-6):
    s = np.sqrt(p * (1 - p))
    return (N - 1) * p + (2 * s + eps) * np.sqrt(N)


if __name__ == "__main__":
    heatmap = np.load(
        "./final_results_sato/results_17_01_2025_sato/sato_heatmap_nagents_15/heatmap_iteration_4_artefact_fixed.npy"
    )
    nP, nT = heatmap.shape
    params = {
        "game_parameters": {
            "n_agents": 15,
            "n_actions": 3,
            "n_iter": 3000,
            "game_type": "sato",
        },
        "network_parameters": {"network_type": "er"},
        "nP": nP,
        "nT": nT,
        "n_expt": 12,
        "p_range": [0.1, 1],
        "T_range": [0.1, 4.5],
        "n_refinements": 4,
    }

    params = ExperimentParameters(**params)

    fig = plot_heatmap(heatmap, params=params)
    p_range = np.linspace(params.p_range[0], params.p_range[1], nP)
    iii = networkgames.compute_iii(
        getattr(networkgames, params.game_parameters.game_type)
    )
    print(iii)
    bound_values = [iii * er_bound(p, params.game_parameters.n_agents) for p in p_range]
    fig.add_trace(go.Scatter(x=bound_values, y=p_range))
    fig.write_html("jjj.html")
