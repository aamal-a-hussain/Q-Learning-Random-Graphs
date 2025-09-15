from pathlib import Path
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
        "game_type",
        type=str,
        default=None,
        help="Game type: Sato, Shapley, Conflict"
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

def create_bound_plot(root_dir, game_type, out_dir):
    results = Path(root_dir).glob("*.npy")
    
    plt.style.use(["science", "ieee"]) 
    plt.rcParams.update({'font.size': 22})

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)

    p_values = []
    plot_data = []

    for path in tqdm(results):
        p = float(path.stem.split('_')[-1].replace('p',''))
        p_values.append(p)
        result = np.load(path, allow_pickle=True).item()
        plot_data.append((p, result))

    plot_data.sort(key=lambda x: x[0])
    p_values.sort()

    cmap = plt.get_cmap('plasma') 
    colors = cmap(np.linspace(0, 1, len(p_values)))

    for idx, (p, result) in enumerate(plot_data):
        line_color = colors[idx]
        
        if np.isclose(p, 0.75):
            line_color = 'green'

        ax.plot(
            list(result.keys()),
            list(result.values()),
            color=line_color,
            label=f"p = {p:.2f}",
            linewidth=3,  # Increased line width
            marker='o',
            markersize=7 # Increased marker size
        )

    ax.legend(title="Edge Probability (p)", loc='upper left', fontsize=16, title_fontsize=18, frameon=True, facecolor='white', edgecolor='black')

    ax.set_title(f"Network {game_type.capitalize()} Game", fontsize=24, pad=20)
    ax.set_xlabel("Number of Agents (N)", fontsize=22)
    ax.set_ylabel("Exploration Rate (T)", fontsize=22)
    
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    fig.savefig(out_dir, bbox_inches="tight", dpi=300)
    print(f"Plot saved to {out_dir}")


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    parser = create_argparser()
    args = parser.parse_args()
    assert isinstance(args.root_dir, str)
    assert isinstance(args.out_dir, str)
    assert isinstance(args.game_type, str)
    create_bound_plot(args.root_dir, args.game_type, args.out_dir)