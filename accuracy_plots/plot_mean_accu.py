import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from NNetwork import NNetwork as nn
from src.sampling import sampling
from accuracy_plots.accu_helper import calculate_degrees
from accu_helper import *



NETWORK_LIST = ['Caltech36', 'UCLA26', 'MIT8', 'Harvard1']

for network in NETWORK_LIST:
    er_accu_all = []
    ba_accu_all = []
    ws_accu_all = []
    gc_accu_all = []

    for iteration in range(1):
        college_graph = nn.NNetwork()
        path = "data/social/" + str(network) + '.txt'
        college_graph.load_add_edges(path, increment_weights=False, use_genfromtxt=True)

        n = len(college_graph.vertices)
        e = len(college_graph.edges)
        er_edgelist = generate_er_graph(n, e)
        ba_edgelist = generate_ba_graph(n, e)
        ws_edgelist = generate_ws_graph(n, e)
        gc_edgelist = generate_config_model(calculate_degrees(path))
        
        er = nn.NNetwork()
        ba = nn.NNetwork()
        ws = nn.NNetwork()
        gc = nn.NNetwork()
        
        er.add_edges(er_edgelist)
        ba.add_edges(ba_edgelist)
        ws.add_edges(ws_edgelist)
        gc.add_edges(gc_edgelist)

        er_accu = []
        ba_accu = []
        ws_accu = []
        gc_accu = []

        for i in range(5, 8):
            print(f"Running k = {i}, Iteration = {iteration + 1}, Network = {network}")
            er_val = similarity(college_graph, network2=er, k=i, n_components=16)
            ba_val = similarity(college_graph, network2=ba, k=i, n_components=16)
            ws_val = similarity(college_graph, network2=ws, k=i, n_components=16)
            gc_val = similarity(college_graph, network2=gc, k=i, n_components=16)
            
            er_accu.append(er_val)
            ba_accu.append(ba_val)
            ws_accu.append(ws_val)
            gc_accu.append(gc_val)

        er_accu_all.append(er_accu)
        ba_accu_all.append(ba_accu)
        ws_accu_all.append(ws_accu)
        gc_accu_all.append(gc_accu)

    # Convert lists to numpy arrays for easier calculations
    er_accu_all = np.array(er_accu_all)
    ba_accu_all = np.array(ba_accu_all)
    ws_accu_all = np.array(ws_accu_all)
    gc_accu_all = np.array(gc_accu_all)

    # Calculate mean and standard deviation
    er_mean = np.mean(er_accu_all, axis=0)
    ba_mean = np.mean(ba_accu_all, axis=0)
    ws_mean = np.mean(ws_accu_all, axis=0)
    gc_mean = np.mean(gc_accu_all, axis=0)

    er_std = np.std(er_accu_all, axis=0)
    ba_std = np.std(ba_accu_all, axis=0)
    ws_std = np.std(ws_accu_all, axis=0)
    gc_std = np.std(gc_accu_all, axis=0)

    # X-axis values (for k from 5 to 19)
    x_values = np.arange(5, 20)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(x_values, er_mean, label='ER', color='blue')
    plt.fill_between(x_values, er_mean - er_std, er_mean + er_std, color='blue', alpha=0.3)

    plt.plot(x_values, ba_mean, label='BA', color='green')
    plt.fill_between(x_values, ba_mean - ba_std, ba_mean + ba_std, color='green', alpha=0.3)

    plt.plot(x_values, ws_mean, label='WS', color='red')
    plt.fill_between(x_values, ws_mean - ws_std, ws_mean + ws_std, color='red', alpha=0.3)

    plt.plot(x_values, gc_mean, label='GC', color='orange')
    plt.fill_between(x_values, gc_mean - gc_std, gc_mean + gc_std, color='orange', alpha=0.3)

    plt.title(f'Comparison of Graph Models for {network}')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(f"Output/accu_plots/Mean-Graph-{network}.png")
    plt.clf()
