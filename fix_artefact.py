# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 23:54:07 2025

@author: dleon
"""

import numpy as np
import plotly.graph_objects as go
from QL_erdos_renyi_create_grid import process_single_cell
import multiprocessing as mp
from tqdm import tqdm
import os
import json


def extract_region(heatmap, p_range, T_range, full_p_range, full_T_range):
    """
    Extract a region from the heatmap based on p and T ranges. This is passed 
    by the user, based on where the artefact is. For Sato and Shapley, this is
    somewhere at the top / top-right.


    Parameters:
    -----------
    heatmap : numpy.ndarray
        The full heatmap array
    p_range : tuple
        (p_min, p_max) for the region of interest, passed by the user.
    T_range : tuple
        (T_min, T_max) for the region of interest, passed by the user.
    full_p_range : tuple
        (p_min, p_max) for the full heatmap, extracted from the json.
    full_T_range : tuple
        (T_min, T_max) for the full heatmap, extracted from the json

    Returns:
    --------
    numpy.ndarray
        The extracted region
    tuple
        The indices used for extraction (p_start, p_end, T_start, T_end)
    """
    nP, nT = heatmap.shape

    # Convert p and T values to indices
    p_indices = np.linspace(full_p_range[0], full_p_range[1], nP)
    T_indices = np.linspace(full_T_range[0], full_T_range[1], nT)

    # Find the closest indices for our region boundaries
    p_start = np.argmin(np.abs(p_indices - p_range[0]))
    p_end = np.argmin(np.abs(p_indices - p_range[1]))
    T_start = np.argmin(np.abs(T_indices - T_range[0]))
    T_end = np.argmin(np.abs(T_indices - T_range[1]))

    # Extract the region
    region = heatmap[p_start:p_end+1, T_start:T_end+1]

    return region, (p_start, p_end, T_start, T_end)


def refine_region(region, n_subdivisions, p_range, T_range, n_agents, n_actions, n_iter, n_expt, game_type):
    """
    Refine a region by subdividing each cell into n_subdivisions x n_subdivisions.
    This is meant to be used to fix the artefact. Assuming only one artefact.

    Parameters:
    -----------
    region : numpy.ndarray
        The region to refine
    n_subdivisions : int
        Number of subdivisions per axis for each cell
    other parameters : same as in process_single_cell

    Returns:
    --------
    numpy.ndarray
        The refined region
    """
    nP, nT = region.shape
    new_nP = nP * n_subdivisions
    new_nT = nT * n_subdivisions

    # Create new parameter grids
    ps = np.linspace(p_range[0], p_range[1], new_nP)
    Ts = np.linspace(T_range[0], T_range[1], new_nT)

    # Initialize refined region
    refined_region = np.zeros((new_nP, new_nT))

    # Create parameter list for all cells
    params_list = []
    for i, p in enumerate(ps):
        for j, T in enumerate(Ts):
            params_list.append(
                (p, T, n_agents, n_actions, n_iter, n_expt, game_type))

    # Process cells in parallel
    num_processes = mp.cpu_count()
    print(
        f"Processing {len(params_list)} cells using {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_cell, params_list),
            total=len(params_list)
        ))

    # Reshape results into refined region
    for idx, result in enumerate(results):
        i = idx // new_nT
        j = idx % new_nT
        refined_region[i, j] = result

    return refined_region


def analyze_region(heatmap_folder_path, heatmap_iteration, p_range, T_range, n_artefact_subdivisions,
                   full_p_range, full_T_range, n_agents, n_actions, n_iter, n_expt, game_type):
    """
    Main function to analyze a specific region of the heatmap. Workflow is

    1) Load existing heatmap in the form of an np.array
    2) Extract the region of the artefact from the user-passed rectangle
    3) Plot the artefact region
    4) Refine the artefact and plot the refinement

    Parameters:
    -----------
    heatmap_path : str
        Path to the full heatmap numpy file
    p_range : tuple
        (p_min, p_max) for region of interest
    T_range : tuple
        (T_min, T_max) for region of interest
    n_artefact_subdivisions : int
        Number of subdivisions per axis for each cell
    other parameters : same as in process_single_cell
    """
    # Load the full heatmap
    heatmap = np.load(os.path.join(heatmap_folder_path,
                      f'heatmap_iteration_{heatmap_iteration}.npy'))

    # Extract the region of interest
    region, indices = extract_region(
        heatmap, p_range, T_range, full_p_range, full_T_range)

    # Plot the original region
    fig = go.Figure(data=go.Heatmap(
        z=region,
        x=np.linspace(T_range[0], T_range[1], region.shape[1]),
        y=np.linspace(p_range[0], p_range[1], region.shape[0])
    ))
    fig.update_layout(
        title="Artefact region",
        xaxis_title="Temperature (T)",
        yaxis_title="Probability (p)"
    )
    fig.write_html("Artefact_region.html")

    # Refine the region
    refined_region = refine_region(
        region,
        n_artefact_subdivisions,
        p_range,
        T_range,
        n_agents,
        n_actions,
        n_iter,
        n_expt,
        game_type
    )

    # Plot the refined region
    fig = go.Figure(data=go.Heatmap(
        z=refined_region,
        x=np.linspace(T_range[0], T_range[1], refined_region.shape[1]),
        y=np.linspace(p_range[0], p_range[1], refined_region.shape[0])
    ))
    fig.update_layout(
        title="Refined Artefact Region",
        xaxis_title="Temperature (T)",
        yaxis_title="Probability (p)"
    )
    fig.write_html(os.path.join(heatmap_folder_path,
                   "refined_Artefact_region.html"))

    # Save the refined region
    np.save(os.path.join(heatmap_folder_path,
            "refined_artefact_region.npy"), refined_region)

    return refined_region


def integrate_refined_region(original_heatmap, refined_region, region_indices, n_subdivisions):
    """
    Integrate a refined region back into the full heatmap, expanding all cells.

    Parameters:
    -----------
    original_heatmap : numpy.ndarray
        The original full heatmap
    refined_region : numpy.ndarray
        The refined region output from analyze_region
    region_indices : tuple
        (p_start, p_end, T_start, T_end) indices from extract_region
    n_subdivisions : int
        Number of subdivisions used in refinement

    Returns:
    --------
    numpy.ndarray
        The complete heatmap with the refined region integrated and all other
        cells expanded to match the resolution
    """
    nP, nT = original_heatmap.shape
    new_nP = nP * n_subdivisions
    new_nT = nT * n_subdivisions

    # Initialize the new full-size heatmap
    new_heatmap = np.zeros((new_nP, new_nT))

    # Extract region indices
    p_start, p_end, T_start, T_end = region_indices

    # Fill in all cells
    for i in range(nP):
        for j in range(nT):
            # Calculate the indices for the expanded cell
            new_i_start = i * n_subdivisions
            new_i_end = (i + 1) * n_subdivisions
            new_j_start = j * n_subdivisions
            new_j_end = (j + 1) * n_subdivisions

            # Check if this cell is part of the refined region
            if (p_start <= i <= p_end) and (T_start <= j <= T_end):
                # Calculate position in refined region
                refined_i = (i - p_start) * n_subdivisions
                refined_j = (j - T_start) * n_subdivisions

                # Copy the refined data
                new_heatmap[new_i_start:new_i_end, new_j_start:new_j_end] = \
                    refined_region[refined_i:refined_i + n_subdivisions,
                                   refined_j:refined_j + n_subdivisions]
            else:
                # Fill with repeated original value
                new_heatmap[new_i_start:new_i_end, new_j_start:new_j_end] = \
                    original_heatmap[i, j]

    return new_heatmap


if __name__ == '__main__':

    #heatmap_folder_path = r'D:/3rd_dyn_of_games_paper/results_17_01_2025_sato_shapley/sato_heatmap_nagents_15'
    heatmap_folder_path = r"/home/leonted/git_QL/Q-Learning-Random-Graphs/sato_heatmap_nagents_15/"
    artefact_p_range = (0.94, 1.)
    artefact_T_range = (1.75, 1.81)
    n_artefact_subdivisions = 2
    n_expt = 5

    def fix_artefact(heatmap_folder_path, artefact_p_range, artefact_T_range,
                     n_artefact_subdivisions, n_expt):

        params_path = os.path.join(heatmap_folder_path, 'parameters.json')

        with open(params_path, 'rb') as file:

            params = json.load(file)

        # Load and extract params
        n_agents = params['n_agents']
        n_actions = params['n_actions']
        # NOT TO BE CONFUSED WITH n_iter in Q learning or n_artefact_subdivisions. this is the # of refinmenets already done
        n_refinements = int(params['n_refinements'])
        full_p_range = params['p_range']
        full_T_range = params['T_range']
        nP = params['nP'] * 2**n_refinements
        nT = params['nT'] * 2**n_refinements
        n_iter = params['n_iter']
        game_type = params['game_type']

        # Sanity check
        original_heatmap_path = os.path.join(
            heatmap_folder_path, f'heatmap_iteration_{n_refinements}.npy')
        original_heatmap = np.load(original_heatmap_path)
        assert original_heatmap.shape == (nP, nT)

        refined_region = analyze_region(
            heatmap_folder_path, n_refinements,
            artefact_p_range, artefact_T_range,
            n_artefact_subdivisions,
            full_p_range, full_T_range,
            n_agents, n_actions, n_iter,
            n_expt, game_type
        )

        _, region_indices = extract_region(
            original_heatmap,
            artefact_p_range, artefact_T_range,
            full_p_range, full_T_range
        )

        complete_refined_heatmap = integrate_refined_region(
            original_heatmap,
            refined_region,
            region_indices,
            n_artefact_subdivisions
        )

        output_path = os.path.join(
            heatmap_folder_path,
            f'heatmap_iteration_{n_refinements}_artefact_fixed.npy'
        )
        np.save(output_path, complete_refined_heatmap)

        # Create and save visualization of complete heatmap
        fig = go.Figure(data=go.Heatmap(
            z=complete_refined_heatmap,
            x=np.linspace(full_T_range[0], full_T_range[1],
                          complete_refined_heatmap.shape[1]),
            y=np.linspace(full_p_range[0], full_p_range[1],
                          complete_refined_heatmap.shape[0])
        ))
        fig.update_layout(
            title="Complete Refined Heatmap with Fixed Artefact",
            xaxis_title="Temperature (T)",
            yaxis_title="Probability (p)"
        )
        fig.write_html(os.path.join(heatmap_folder_path,
                       "complete_refined_heatmap.html"))

    fix_artefact(heatmap_folder_path, artefact_p_range, artefact_T_range,
                 n_artefact_subdivisions, n_expt)
