import numpy as np
import seaborn as sns
import sys
import os
import matplotlib.pyplot as plt
import pickle  # To save and load dictionaries
from itertools import combinations
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from NNetwork import NNetwork as nn
from contextlib import contextmanager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sampling.Sampling import sampling_sndl
from src.supervised_NDL.SNDL import sndl_equalEdge, sndl_predict

@contextmanager
def suppress_output():
    # Redirect stdout to null
    with open(os.devnull, 'w') as fnull:
        original_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = original_stdout
# Binary 
def compute_latent_motifs_binary_all(graph_list, sample_size_list, k, xi, n_components, iterations, skip_folded_hom):
    motifs = {}
    for i, j in combinations(range(len(graph_list)), 2):
        print(f"Computing latent motifs for networks ({i}, {j})")
        X, y = sampling_sndl([graph_list[i], graph_list[j]], k=k, sample_size_list=sample_size_list, skip_folded_hom=skip_folded_hom)
        with suppress_output():
            W, beta, H = sndl_equalEdge([graph_list[i], graph_list[j]], sample_size_list, k=k, xi=xi, 
                                        n_components=n_components, iter=iterations, skip_folded_hom=skip_folded_hom)
        motifs[(i, j)] = (W, beta)
    return motifs

def compute_affinity_scores(motifs, graph_paths):
    affinity_scores = {}
    num_graphs = len(graph_paths)
    
    for (i, j), (W, beta) in motifs.items():
        for l in range(num_graphs):
            print(f"Computing affinity score for pair ({i}, {j}) with test network {l}")
            G_test = nn.NNetwork()
            G_test.load_add_edges(graph_paths[l], increment_weights=False, use_genfromtxt=True)
            affinity_score = sndl_predict(G_test, W, beta, 1000)
            affinity_scores[(i, j, l)] = affinity_score
            del G_test  # Clear memory after usage
    return affinity_scores

def plot_affinity_heatmap_binary_all(affinity_scores, ntwk_list):
    num_graphs = len(ntwk_list)
    num_pairs = len(list(combinations(range(num_graphs), 2)))
    affinity_matrix = np.zeros((num_pairs, num_graphs))
    
    row_labels = []
    idx = 0
    
    for i, j in combinations(range(num_graphs), 2):
        row_labels.append(f'{ntwk_list[i]} & {ntwk_list[j]}')
        for l in range(num_graphs):
            affinity_matrix[idx, l] = affinity_scores[(i, j, l)][1]
        idx += 1

    col_labels = [ntwk for ntwk in ntwk_list]

    plt.figure(figsize=(10, 8))
    sns.heatmap(affinity_matrix, annot=True, fmt=".2f", xticklabels=col_labels, yticklabels=row_labels, cmap='Blues')
    plt.xlabel('Test Network')
    plt.ylabel('Network Pair')
    plt.title('Affinity Scores Heatmap')
    plt.show()

#Main function for Binary
def affinity_analysis_binary_all(ntwk_list, sample_size_list, k, xi, n_components, iterations, skip_folded_hom):
    graph_paths = [f"data/{ntwk}.txt" for ntwk in ntwk_list]
    graph_list = []

    for path in graph_paths:
        G = nn.NNetwork()
        G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
        graph_list.append(G)
    
    motifs = compute_latent_motifs_binary_all(graph_list, sample_size_list, k, xi, n_components, iterations, skip_folded_hom)
    affinity_scores = compute_affinity_scores(motifs, graph_paths)
    plot_affinity_heatmap_binary_all(affinity_scores, ntwk_list)



##### Multiclass

def get_save_path(ntwk_list, base_sample_size, k, xi, n_components, iterations, baseline_i, average=False):
    """Generate a unique file path for saving the dictionary based on parameters."""
    params_str = f"bs{base_sample_size}_k{k}_xi{xi}_nc{n_components}_iter{iterations}_bi{baseline_i}"
    if average:
        params_str += "_avg"
    ntwk_str = "_".join(ntwk_list[baseline_i:baseline_i+3])
    filename = f"dictionaries/{ntwk_str}_{params_str}.pkl"
    print("Getting save path...")
    return filename

def save_dictionary(W, beta, H, filepath):
    """Save W, beta, H to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump((W, beta, H), f)
        print("Saving dictionary...")

def load_dictionary(filepath):
    """Load W, beta, H from a file."""
    with open(filepath, 'rb') as f:
        W, beta, H = pickle.load(f)
        print("Loading dictionary...")
    return W, beta, H

def compute_latent_motifs_and_dictionary(ntwk_list, base_sample_size, k, xi, n_components, iterations, baseline_i=0, skip_folded_hom=True, average=False, times=1):
    """Compute or load the latent motifs and dictionary for a given baseline."""
    filepath = get_save_path(ntwk_list, base_sample_size, k, xi, n_components, iterations, baseline_i, average=average)
    
    # Only load precomputed dictionary if not averaging
    if os.path.exists(filepath) and not average:
        print(f"Loading precomputed dictionary from {filepath}")
        W, beta, H = load_dictionary(filepath)
    else:
        print(f"Computing dictionary for baseline {baseline_i} and saving to {filepath}")
        graph_list = []
        for ntwk in ntwk_list[baseline_i:baseline_i+3]:
            path = f"data/{ntwk}.txt"
            G = nn.NNetwork()
            G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
            graph_list.append(G)
            print(f"Calling `sndl_equalEdge` and computing dictionary for {ntwk}")

        for trial in range(times):
            with suppress_output():
                W, beta, H = sndl_equalEdge(graph_list, base_sample_size=base_sample_size, k=k, xi=xi, n_components=n_components, iter=iterations, skip_folded_hom=skip_folded_hom)
            
            # Save only the last computed dictionary
            if trial == times - 1:  # Save the last trial's dictionary
                save_dictionary(W, beta, H, filepath)
    
    return W, beta, H

def compute_prediction_scores(G, W, beta, sample_size):
    """Compute prediction scores for a given graph and dictionary."""
    with suppress_output():
        prob = sndl_predict(G, W, beta, sample_size)
        print("Computing prediction scores...")
    return prob

def compute_affinity_scores_for_all_networks(ntwk_list, W, beta, average=False, times=1):
    """Compute affinity scores for the networks, with optional averaging over multiple trials."""
    affinity_scores = {ntwk: [] for ntwk in ntwk_list}

    for trial in range(times):
        for ntwk in ntwk_list:
            path = "data/" + str(ntwk) + '.txt'
            G = nn.NNetwork()
            G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
            prob = compute_prediction_scores(G, W, beta, 500)
            affinity_scores[ntwk].append(prob)

    if average:
        avg_affinity_scores = {}
        std_affinity_scores = {}
        for ntwk in ntwk_list:
            scores = np.array(affinity_scores[ntwk])
            avg_affinity_scores[ntwk] = np.mean(scores, axis=0)
            std_affinity_scores[ntwk] = np.std(scores, axis=0)
        return avg_affinity_scores, std_affinity_scores
    else:
        # Return the first trial's result if not averaging
        return affinity_scores


def plot_3d_affinity_scores(ntwk_list, affinity_scores, baseline_i, view_angle=(30, 60), average=False, times=1, ax=None):
    """Plot 3D affinity scores for the networks on a single graph, with option to include standard deviation."""
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = plt.gcf()  # Get the current figure if using an existing subplot

    print("Plotting...")

    colormap = plt.colormaps.get_cmap('tab10')
    colors = [colormap(i / len(ntwk_list)) for i in range(len(ntwk_list))]
    markers = ['v', 's', 'p', '^', 'D', '*', 'P', 'X', 'h']

    # Plot Standards with the same color as the first three networks
    standard_points = []
    for idx in range(3):
        point = [idx == 0, idx == 1, idx == 2]
        standard_points.append(point)
        ax.scatter(point[0], point[1], point[2], color=colors[baseline_i + idx], s=50)

    # Plot prediction scores for all networks    
    all_points = []
    labels = []
    coordinates = []
    std_devs = []
    
    for idx, ntwk in enumerate(ntwk_list):
        if average:
            prob = affinity_scores[0][ntwk]
        else:
            prob = affinity_scores[ntwk][0]
            
        std_dev = affinity_scores[1][ntwk] if average else [0, 0, 0]

        point = np.array([prob[0], prob[1], prob[2]])
        all_points.append(point)
        labels.append(ntwk)
        coordinates.append(f'({prob[0]:.2f}, {prob[1]:.2f}, {prob[2]:.2f})')
        std_devs.append(f'({std_dev[0]:.2f}, {std_dev[1]:.2f}, {std_dev[2]:.2f})')

        if idx in [baseline_i, baseline_i+1, baseline_i+2]:
            ax.scatter(point[0], point[1], point[2], color=colors[idx], s=50, label=f'{ntwk}', marker='o')
        else:
            marker_idx = (idx + 3) % len(markers)  # Use a different marker shape for non-baseline points
            ax.scatter(point[0], point[1], point[2], color=colors[idx], s=50, marker=markers[marker_idx], label=f'{ntwk}')

    tri_vertices = np.array(standard_points)
    tri = Poly3DCollection([tri_vertices], alpha=0.3, color='grey')
    ax.add_collection3d(tri)

    small_tri_vertices = np.array(all_points[baseline_i:baseline_i+3])
    small_tri = Poly3DCollection([small_tri_vertices], alpha=0.3, edgecolor='r', color='yellow')
    ax.add_collection3d(small_tri)

    def triangle_area(p1, p2, p3):
        p1, p2, p3 = np.array(p1, dtype=float), np.array(p2, dtype=float), np.array(p3, dtype=float)
        return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

    big_triangle_area = triangle_area(*tri_vertices)
    small_triangle_area = triangle_area(*small_tri_vertices[:3])
    area_ratio = small_triangle_area / big_triangle_area

    ax.text2D(0.05, 0.95, f'Area Ratio = {small_triangle_area:.2f} / {big_triangle_area:.2f} = {area_ratio:.2f}', transform=ax.transAxes)
    
    ax.set_xlabel(ntwk_list[baseline_i])
    ax.set_ylabel(ntwk_list[baseline_i+1])
    ax.set_zlabel(ntwk_list[baseline_i+2])
    if not average:
        ax.set_title('3D Visualization of Predicted Network Similarities')
    else:
        ax.set_title(f'3D Visualization of Average Predicted Network Similarities ({times} times)')

    # Only add the legend and table if we're not in a subplot (to avoid overcrowding)
    if ax.get_subplotspec() is None:
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), ncol=1)
    
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    return fig if ax.get_subplotspec() is None else None  # Return the figure if it's not a subplot


def save_plot(fig, ntwk_list, base_sample_size, k, n_components, iterations, baseline_i, average):
    """Save the plot to a file with a concise name including network names and parameters."""
    output_dir = "output/triangle_plot"
    os.makedirs(output_dir, exist_ok=True)

    # Construct the filename
    ntwk_str = "_".join(ntwk_list[:3]) + "_etc" if len(ntwk_list) > 3 else "_".join(ntwk_list)
    avg_str = "avg" if average else "single"
    filename = f"{ntwk_str}_bs{base_sample_size}_k{k}_nc{n_components}_iter{iterations}_bi{baseline_i}_{avg_str}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save the figure
    fig.savefig(filepath)
    print(f"Plot saved to {filepath}")

# Multiclass Main Function 

def plot_3d_prediction(ntwk_list, base_sample_size, k, xi, n_components, iterations, baseline_i, skip_folded_hom=True, average=False, times=1, ax=None):
    """Compute and plot 3D prediction of network similarities."""
    n = len(ntwk_list)
    if n <= 3:
        print("Not enough networks for comparison.")
        return

    # Create a directory to store dictionaries if it doesn't exist
    os.makedirs('dictionaries', exist_ok=True)
    
    # Compute dictionary for the baseline set of 3 networks, with optional averaging
    W, beta, H = compute_latent_motifs_and_dictionary(
        ntwk_list, base_sample_size=base_sample_size, k=k, xi=xi, n_components=n_components,
        iterations=iterations, baseline_i=baseline_i, skip_folded_hom=skip_folded_hom,
        average=average, times=times
    )

    # Compute affinity scores for all networks, with optional averaging
    affinity_scores = compute_affinity_scores_for_all_networks(
        ntwk_list, W, beta, average=average, times=times
    )

    # Generate and show a single plot with all networks, including the baseline and additional ones
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_3d_affinity_scores(ntwk_list, affinity_scores, baseline_i, average=average, times=times, ax=ax)
        plt.show()
        return fig  # Return the figure if created
    else:
        # Use the provided ax for plotting, no need to show or return fig
        plot_3d_affinity_scores(ntwk_list, affinity_scores, baseline_i, average=average, times=times, ax=ax)



# Four-baseline Radar Plot
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        original_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = original_stdout

# Dictionary Functions for Four Baselines
def get_save_path_4_baseline(ntwk_list, base_sample_size, k, xi, n_components, iterations, baseline_i, average=False):
    """Generate a unique file path for saving the dictionary based on parameters for four baselines."""
    params_str = f"bs{base_sample_size}_k{k}_xi{xi}_nc{n_components}_iter{iterations}_bi{baseline_i}"
    if average:
        params_str += "_avg"
    ntwk_str = "_".join(ntwk_list[baseline_i:baseline_i+4])
    filename = f"dictionaries/{ntwk_str}_{params_str}.pkl"
    return filename

def save_dictionary(W, beta, H, filepath):
    """Save W, beta, H to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump((W, beta, H), f)

def load_dictionary(filepath):
    """Load W, beta, H from a file."""
    with open(filepath, 'rb') as f:
        W, beta, H = pickle.load(f)
    return W, beta, H

def compute_latent_motifs_and_dictionary_4_baseline(ntwk_list, base_sample_size, k, xi, n_components, iterations, baseline_i=0, skip_folded_hom=True, average=False, times=1):
    """Compute or load the latent motifs and dictionary for four baselines."""
    filepath = get_save_path_4_baseline(ntwk_list, base_sample_size, k, xi, n_components, iterations, baseline_i, average=average)
    
    if os.path.exists(filepath) and not average:
        W, beta, H = load_dictionary(filepath)
    else:
        graph_list = []
        for ntwk in ntwk_list[baseline_i:baseline_i+4]:
            path = f"data/{ntwk}.txt"
            G = nn.NNetwork()  # Correct instantiation
            G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
            graph_list.append(G)

        for trial in range(times):
            with suppress_output():
                W, beta, H = sndl_equalEdge(graph_list, base_sample_size=base_sample_size, k=k, xi=xi, n_components=n_components, iter=iterations, skip_folded_hom=skip_folded_hom)
            if trial == times - 1:
                save_dictionary(W, beta, H, filepath)
    
    return W, beta, H

def compute_affinity_scores_for_all_networks_4_baseline(ntwk_list, W, beta, average=False, times=1):
    """Compute affinity scores for all networks using four baseline networks."""
    num_baselines = 4
    affinity_scores = {ntwk: [0] * num_baselines for ntwk in ntwk_list}

    for trial in range(times):
        for idx, ntwk in enumerate(ntwk_list):
            path = f"data/{ntwk}.txt"
            G = nn.NNetwork()  # Corrected instantiation of NNetwork
            G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
            prob = compute_prediction_scores(G, W, beta, 500)
            
            # Ensure prob has the correct number of elements
            if len(prob) != num_baselines:
                raise ValueError(f"Expected {num_baselines} elements in affinity scores, but got {len(prob)}")

            # Accumulate scores for averaging
            for model in range(num_baselines):
                affinity_scores[ntwk][model] += prob[model] / times

    return affinity_scores


# Radar Plot Function
def plot_radar_affinity_scores(ntwk_list, affinity_scores, title="Radar Plot of Affinity Scores for Synthetic Networks"):
    """Generate a radar plot to visualize the affinity scores of each network with respect to four baseline types."""
    labels = ['ER', 'WS', 'BA', 'CM']
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for ntwk in ntwk_list:
        values = affinity_scores[ntwk]
        values = np.append(values, values[0])

        ax.plot(angles, values, label=ntwk, linewidth=2)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.set_yticklabels(["0.2", "0.5", "0.8"], color="grey", size=8)
    ax.set_ylim(0, 1)

    plt.title(title, size=15, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()
