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
from QL_erdos_renyi_refine_grid import (
    create_refined_heatmap,
    identify_cells_of_interest,
)
from parameters import ExperimentParameters


def create_experiment_folder(parameters: ExperimentParameters):
    """Create folder and save parameters"""
    #folder_name = f"heatmap_nagents_{parameters.game_parameters.n_agents}"
    
    game_parameters = parameters.game_parameters
    network_parameters = parameters.network_parameters
    network_type = network_parameters.network_type
    
    folder_name = f'heatmap_nagents_{game_parameters.n_agents}_game_type_{game_parameters.game_type}_refin_{parameters.n_refinements}_network_type_{network_type}'#_{int(10 * parameters.network_parameters.q)}"
    
    if network_type == 'sbm':
        
        q = network_parameters.q

        
        folder_name += f'_q_{int(10*q)}'
    
    os.makedirs(folder_name, exist_ok=True)

    # Save parameters to JSON file
    params_file = os.path.join(folder_name, "parameters.json")
    with open(params_file, "w") as f:
        json.dump(json.loads(parameters.model_dump_json()), f, indent=4)

    return folder_name


def save_heatmap(heatmap, folder_name, iteration, p_range, T_range):
    """Save heatmap as both .npy and .html"""
    # Save numpy array
    np_filename = os.path.join(folder_name, f"heatmap_iteration_{iteration}.npy")
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
        yaxis_title="Probability (p)",
    )

    html_filename = os.path.join(folder_name, f"heatmap_iteration_{iteration}.html")
    fig.write_html(html_filename)


def run_heatmap_workflow(params: ExperimentParameters):
    """Run the complete heatmap generation and refinement workflow"""

    # Create folder
    folder_name = create_experiment_folder(params)

    # Generate initial heatmap
    print("Generating initial heatmap...")
    heatmap = generate_heatmap(params)

    # Save initial heatmap
    save_heatmap(heatmap, folder_name, 0, params.p_range, params.T_range)

    # Perform refinements
    current_heatmap = heatmap
    current_nP = params.nP
    current_nT = params.nT

    for i in range(params.n_refinements):
        print(f"Performing refinement {i+1}/{params.n_refinements}...")

        if i == params.n_refinements - 1:
            params.n_expt = 50

        # Identify cells of interest
        mask = identify_cells_of_interest(current_heatmap)
        params.nP = current_nP
        params.nT = current_nT

        # Refine heatmap
        current_heatmap = create_refined_heatmap(
            current_heatmap,
            mask,
            params,
        )

        # Update current grid sizes
        current_nP *= 2
        current_nT *= 2

        # Save refined heatmap
        save_heatmap(
            current_heatmap, folder_name, i + 1, params.p_range, params.T_range
        )

    print(f"Workflow completed. Results saved in {folder_name}/")


def main():
    game_parameters = {
        "game_type": "conflict",
        "n_agents": 25,
        "n_actions": 3,
        "n_iter": 4000,
    }
    network_parameters = {"network_type": "er"}#{"network_type": "sbm", "q": 0.1, "n_blocks": 3} #
    params = {
        "game_parameters": game_parameters,
        "network_parameters": network_parameters,
        "nP": 64,
        "nT": 64,
        "n_expt": 12,
        "p_range": (0.05, .25),
        "T_range": (0.05, 4.25), # (0.05,4.25) is a good range for this
        "n_refinements": 3,
    }

    parameters = ExperimentParameters(**params)
    run_heatmap_workflow(parameters)


if __name__ == "__main__":
    main()
