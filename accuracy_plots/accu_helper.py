import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from NNetwork import NNetwork as nn
from src.sampling import sampling
import src.SMF_BCD as SMF

NETWORK_LIST = ['Caltech36', 'UCLA26', 'MIT8', 'Harvard1']

def generate_er_graph(n, e):
    m = 2*n/e
    er_graph = nx.erdos_renyi_graph(n, m)
    edgelist = list(er_graph.edges())
    return edgelist

def generate_ba_graph(n, e):
    m = 2*n/e
    p = int(m*(n-1)/2)
    ba_graph = nx.barabasi_albert_graph(n, p)
    edgelist = list(ba_graph.edges())
    return edgelist

def generate_ws_graph(n, e):
    d = 2*n/e
    k = int(d*(n-1))
    p = 0.5
    ws_graph = nx.watts_strogatz_graph(n, k, p)
    edgelist = list(ws_graph.edges())
    return edgelist

def generate_config_model(degree_sequence):
    config_graph = nx.configuration_model(degree_sequence)
    simple_graph = nx.Graph(config_graph)  # Removes parallel edges
    simple_graph.remove_edges_from(nx.selfloop_edges(simple_graph))
    edgelist = list(simple_graph.edges())
    return edgelist

def similarity(network1, network2, k=30, xi=5, n_components=8):
    Output = live_sample_graph(network1, network2, k=k)
    X = Output[0]
    Y = Output[1][:, np.newaxis]
    Xtrain, Ytrain = X, Y
    SMF_Train = SMF.SDL_BCD([X, Y.T], X_test=[Xtrain, Ytrain.T], xi=xi, n_components=n_components)
    results_dict_new = SMF_Train.fit(iter=250, subsample_size=None, option="filter", if_compute_recons_error=True, if_validate=True)
    W = results_dict_new.get('loading')[0]
    beta = results_dict_new.get('loading')[1]
    threshold = results_dict_new.get('Opt_threshold')
    prediction1 = live_sample_graph(network1, network1, k=k)[0]
    pred = []
    y = []
    correct = 0
    for j in range(prediction1.shape[0]):
        a = beta[:, 1:] @ W.T @ prediction1[:, j] + beta[:, 0]
        if a < threshold:
            correct += 1
            pred.append(0)
        else:
            pred.append(1)
        y.append(0)
    
    prediction2 = live_sample_graph(network2, network2, k=k)[0]
    
    for j in range(prediction2.shape[0]):
        a = beta[:, 1:] @ W.T @ prediction2[:, j] + beta[:, 0]
        if a < threshold:
            pred.append(0)
        else:
            pred.append(1)
            correct += 1
        # pred.append(1 / (1 + np.exp(-a)[0]))
        y.append(1)
    
    return correct/len(y)

def live_sample_graph(graph1, graph2, k=30, tolerance=0.05):
    graph_list = [graph1, graph2]
    if graph1 == graph2:
        return sampling(graph_list, k=k, sample_size_list =[500,500])
    guess1, guess2 = 500, 500
    A, y = sampling(graph_list, k=k, sample_size_list=[guess1, guess2])
    graph1_val = np.sum(A[:, :guess1])
    graph2_val = np.sum(A[:, guess1 + 1:])
    if graph1_val > graph2_val:
        difference = graph1_val - graph2_val
        graph2_unit = graph2_val/500
        guess2 += difference//graph2_unit
    else:
        difference = graph2_val - graph1_val
        graph1_unit = graph1_val/500
        guess1 += difference//graph1_unit
    A, y = sampling(graph_list, k=k, sample_size_list=[int(guess1), int(guess2)])
    return A, y, guess1, guess2

def calculate_degrees(path):
    from collections import defaultdict
    degrees = defaultdict(int)
    with open(path, 'r') as file:
        for line in file:
            x, y = line.strip().split(',')
            degrees[x] += 1
            degrees[y] += 1
    degrees = dict(sorted(degrees.items()))
    degrees = list(degrees.values())
    return degrees