import json
import os
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm

from parameters import ExperimentParameters
from scripts import GAME_REGISTRY


def create_argparser():
    # ... (same as before)
    parser = ArgumentParser()
    parser.add_argument(
        "root_dir",
        type=str,
        default=None,
        help="Root directory containing experiment folders.",
    )
    parser.add_argument(
        "out_dir",
        type=str,
        default=None,
        help="Output directory to save the plot.",
    )
    parser.add_argument(
        "--plot_bound",
        action="store_true",
        help="Whether to plot the theoretical boundary.",
    )
    parser.add_argument(
        "--plot_cbar",
        action="store_true",
        help="Whether to plot the colour bar.",
    )
    return parser


def load_parameters(root_path, folder):
    # ... (same as before)
    with open(os.path.join(root_path, folder, "parameters.json"), "r") as f:
        params = json.load(f)
    params = ExperimentParameters(**params)
    return params


def create_heatmap_subplots(root_path: str, plot_bound: bool, plot_cbar: bool, out_dir: str):
    # Apply scienceplots style
    plt.style.use("science")

    plt.rcParams.update({'font.size': 24})  # Global font size
    
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(6 * 4, 6 * 3),
        dpi=300,
        constrained_layout=True,
    )
    for row, game_type in enumerate(["sato", "shapley", "conflict"]):
        game_path = os.path.join(root_path, f"{game_type}_er")
        subdir = sorted(os.listdir(game_path), key=lambda x: int(x.split("_")[-1]))
        for col, folder in tqdm(enumerate(subdir)):
            ax = axes[row, col]
            heatmap_npy = np.load(os.path.join(game_path, folder, "heatmap_iteration_3.npy"))
            params = load_parameters(game_path, folder)
            n_agents = params.game_parameters.n_agents

            hmp = plot_heatmap(ax, heatmap_npy, params)
            if plot_bound:
                plot_heatmap_and_bound(ax, heatmap_npy, params)

            # Update subplot titles and labels with larger font size
            if row == 0:
                if col == 0:
                    ax.set_title(f"Network {game_type.capitalize()} \n N = {n_agents}", fontsize=32)
                else:
                    ax.set_title(f"N = {n_agents}", fontsize=32)
            else:
                if col == 0:
                    ax.set_title(f"Network {game_type.capitalize()}", fontsize=32)

            if col == 0:
                ax.set_ylabel("Edge Probability (p)", fontsize=28)
            
            if row == 2:
                ax.set_xlabel("Exploration Rate (T)", fontsize=28)
            
            # Make tick labels larger
            ax.tick_params(axis='both', which='major', labelsize=24)

    if plot_cbar:
        cbar = fig.colorbar(hmp, ax=axes, shrink=0.7, orientation="vertical")
        cbar.ax.tick_params(labelsize=24)
        cbar.set_label("Value", fontsize=28)

    fig.savefig(out_dir, bbox_inches="tight")
    print(f"Plot saved to {out_dir}")


def plot_heatmap(ax, heatmap, params: ExperimentParameters):
    # ... (same as before)
    nP, nT = heatmap.shape
    p_range = np.linspace(params.p_range[0], params.p_range[1], nP)
    T_range = np.linspace(params.T_range[0], params.T_range[1], nT)
    hmp = ax.imshow(
        heatmap,
        cmap="cividis",
        aspect="auto",
        origin="lower",
        extent=[T_range[0], T_range[-1], p_range[0], p_range[-1]],
    )
    return hmp


def plot_heatmap_and_bound(ax, heatmap, params: ExperimentParameters, upper_bound=6.5):
    # ... (same as before)
    def er_bound(p, N, eps=1e-6):
        s = np.sqrt(p * (1 - p))
        return (N - 1) * p + (2 * s + eps) * np.sqrt(N)
    
    nP, nT = heatmap.shape
    p_range = np.linspace(params.p_range[0], params.p_range[1], nP)
    game = GAME_REGISTRY[params.game_parameters.game_type](n_agents=3)
    iii = game.iii
    bound_values = [iii * er_bound(p, params.game_parameters.n_agents) for p in p_range]
    
    ax.plot(
        bound_values,
        p_range,
        color="white",
        linestyle="--",
        linewidth=2,
        label="Boundary",
    )


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    parser = create_argparser()
    args = parser.parse_args()
    assert isinstance(args.root_dir, str)
    assert isinstance(args.out_dir, str)
    create_heatmap_subplots(args.root_dir, args.plot_bound, args.plot_cbar, args.out_dir)