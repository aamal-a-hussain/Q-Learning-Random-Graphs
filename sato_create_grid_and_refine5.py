# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:44:53 2024

@author: dleon
"""

import os
import numpy as np
import json
import plotly.graph_objects as go
from datetime import datetime
from QL_erdos_renyi_create_grid import generate_heatmap  # , plot_heatmap
from QL_erdos_renyi_refine_grid import create_refined_heatmap, identify_cells_of_interest


def create_experiment_folder(n_agents, params):
    """Create folder and save parameters"""
    folder_name = f"heatmap_nagents_{n_agents}"
    os.makedirs(folder_name, exist_ok=True)

    # Save parameters to JSON file
    params_file = os.path.join(folder_name, "parameters.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=4)

    return folder_name


def save_heatmap(heatmap, folder_name, iteration, p_range, T_range):
    """Save heatmap as both .npy and .html"""
    # Save numpy array
    np_filename = os.path.join(
        folder_name, f"heatmap_iteration_{iteration}.npy")
    np.save(np_filename, heatmap)

    # Create and save plotly figure
    fig = go.Figure()
    nP, nT = heatmap.shape
    Ts = np.linspace(T_range[0], T_range[1], nT)
    ps = np.linspace(p_range[0], p_range[1], nP)

    fig.add_trace(go.Heatmap(x=Ts, y=ps, z=heatmap))
    fig.update_layout(
        title=f"Q-Learning Convergence Heatmap (Iteration {iteration})",
        xaxis_title="Temperature (T)",
        yaxis_title="Probability (p)"
    )

    html_filename = os.path.join(
        folder_name, f"heatmap_iteration_{iteration}.html")
    fig.write_html(html_filename)


def run_heatmap_workflow(n_agents, n_actions, nP, nT, n_iter, n_expt,
                         p_range, T_range, game_type, n_refinements):
    """Run the complete heatmap generation and refinement workflow"""

    # Store parameters
    params = {
        "n_agents": n_agents,
        "n_actions": n_actions,
        "nP": nP,
        "nT": nT,
        "n_iter": n_iter,
        "n_expt": n_expt,
        "p_range": p_range,
        "T_range": T_range,
        "game_type": game_type,
        "n_refinements": n_refinements,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }

    # Create folder
    folder_name = create_experiment_folder(n_agents, params)

    # Generate initial heatmap
    print("Generating initial heatmap...")
    heatmap = generate_heatmap(
        n_agents=n_agents,
        n_actions=n_actions,
        nP=nP,
        nT=nT,
        n_iter=n_iter,
        n_expt=n_expt,
        p_range=p_range,
        T_range=T_range,
        game_type=game_type
    )

    # Save initial heatmap
    save_heatmap(heatmap, folder_name, 0, p_range, T_range)

    # Perform refinements
    current_heatmap = heatmap
    current_nP = nP
    current_nT = nT

    for i in range(n_refinements):
        print(f"Performing refinement {i+1}/{n_refinements}...")

        if i == n_refinements - 1 and i > 0:

            n_expt = 50

        # Identify cells of interest
        mask = identify_cells_of_interest(current_heatmap)

        # Refine heatmap
        current_heatmap = create_refined_heatmap(
            current_heatmap,
            mask,
            n_agents=n_agents,
            n_actions=n_actions,
            nP=current_nP,
            nT=current_nT,
            n_iter=n_iter,
            n_expt=n_expt,
            p_range=p_range,
            T_range=T_range,
            game_type=game_type
        )

        # Update current grid sizes
        current_nP *= 2
        current_nT *= 2

        # Save refined heatmap
        save_heatmap(current_heatmap, folder_name, i+1, p_range, T_range)

    print(f"Workflow completed. Results saved in {folder_name}/")


def main():

    params = {
        "n_agents": 5,
        "n_actions": 3,
        "nP": 30,
        "nT": 30,
        "n_iter": 3000,
        "n_expt": 12,
        "p_range": (0.1, 1),
        "T_range": (0.1, 6),
        "game_type": "sato",
        "n_refinements": 0
    }

    run_heatmap_workflow(**params)


if __name__ == "__main__":
    main()
