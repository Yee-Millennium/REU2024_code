import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from NNetwork import NNetwork as nn
from src.sampling.Sampling import sampling_sndl
from src.supervised_NDL.SNDL import sndl_equalEdge, sndl_predict
from util.plotting import *
import os
import sys
from contextlib import contextmanager


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




def compute_latent_motifs_binary_all(graph_list, sample_size_list, k, n_components, iterations):
    motifs = {}
    for i in range(len(graph_list)):
        for j in range(len(graph_list)):
            if i != j:
                print(f"Computing latent motifs for networks ({i}, {j})")
                X, y = sampling_sndl([graph_list[i], graph_list[j]], k=k, sample_size_list=sample_size_list)
                with suppress_output():
                    W, beta, H = sndl_equalEdge([graph_list[i], graph_list[j]], sample_size_1=sample_size_list[0], sample_size_2=sample_size_list[1], k=k, xi=2, n_components=n_components, iter=iterations)
                motifs[(i, j)] = (W, beta)
    return motifs



def plot_affinity_heatmap_binary_all(affinity_scores, ntwk_list):
    num_graphs = len(ntwk_list)
    affinity_matrix = np.zeros((num_graphs * (num_graphs - 1), num_graphs))
    
    row_labels = []
    idx = 0
    
    for i in range(num_graphs):
        for j in range(num_graphs):
            if i != j:
                row_labels.append(f'{ntwk_list[i]} & {ntwk_list[j]}')
                for l in range(num_graphs):
                    affinity_matrix[idx, l] = affinity_scores[(i, j, l)]
                idx += 1

    col_labels = [ntwk for ntwk in ntwk_list]

    plt.figure(figsize=(10, 8))
    sns.heatmap(affinity_matrix, annot=True, fmt=".2f", xticklabels=col_labels, yticklabels=row_labels, cmap='Blues')
    plt.xlabel('Test Network')
    plt.ylabel('Network Pair')
    plt.title('Affinity Scores Heatmap')
    plt.show()



def affinity_analysis_binary_all(ntwk_list, sample_size_list, k, n_components, iterations):
    graph_paths = [f"data/{ntwk}.txt" for ntwk in ntwk_list]
    graph_list = []

    for path in graph_paths:
        G = nn.NNetwork()
        G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
        graph_list.append(G)
    
    motifs = compute_latent_motifs_binary_all(graph_list, sample_size_list, k, n_components, iterations)
    affinity_scores = compute_affinity_scores(motifs, graph_paths, sample_size_list, k, n_components, iterations)
    plot_affinity_heatmap_binary_all(affinity_scores, ntwk_list)