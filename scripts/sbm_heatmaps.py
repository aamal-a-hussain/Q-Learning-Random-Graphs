import json
import os
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm

from parameters import ExperimentParameters


def create_argparser():
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
        help="Output file path for the plot (e.g., plots/output.pdf).",
    )
    return parser


def load_parameters(root_path, folder):
    with open(os.path.join(root_path, folder, "parameters.json"), "r") as f:
        params = json.load(f)
    params = ExperimentParameters(**params)
    return params


def create_heatmap_subplots(root_dir, out_dir):
    Qs = {"1": 0.1, "2": 0.2}
    Ns = [25, 50, 75, 100]

    plt.style.use(["science", "ieee"])
    plt.rcParams.update({'font.size': 16})

    fig, axes = plt.subplots(
        len(Qs),
        len(Ns),
        figsize=(16, 8),
        constrained_layout=True,
    )
    
    mappable = plt.cm.ScalarMappable(cmap='cividis')

    for row, q_key in tqdm(enumerate(Qs.keys())):
        for col, N in tqdm(enumerate(Ns)):
            ax = axes[row, col]
            
            heatmap_path = os.path.join(
                root_dir,
                f"heatmap_nagents_{N}_q_{q_key}",
                "heatmap_iteration_3.npy",
            )
            
            heatmap_npy = np.load(heatmap_path)
            params = load_parameters(root_dir, f"heatmap_nagents_{N}_q_{q_key}")

            hmp = plot_heatmap(ax, heatmap_npy, params)
            
            ax.set_title(f"N = {N}, q = {Qs[q_key]}", fontsize=18)
            
            if col == 0:
                ax.set_ylabel("Edge Probability (p)", fontsize=18)
            
            if row == len(Qs) - 1:
                ax.set_xlabel("Exploration Rate (T)", fontsize=18)

            ax.tick_params(axis='both', which='major', labelsize=14)

    cbar = fig.colorbar(hmp, ax=axes, orientation='vertical', shrink=0.8)
    cbar.ax.tick_params(labelsize=20)  
    cbar.set_label("Value", fontsize=22) 

    fig.savefig(out_dir)
    print(f"Plot saved to {out_dir}")


def plot_heatmap(ax, heatmap, params: ExperimentParameters):
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


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    parser = create_argparser()
    args = parser.parse_args()
    assert isinstance(args.root_dir, str)
    assert isinstance(args.out_dir, str)
    create_heatmap_subplots(args.root_dir, args.out_dir)