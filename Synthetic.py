import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.visualization import *
from NNetwork import NNetwork as nn
from src.supervised_NDL.SNDL import sndl_equalEdge, sndl_predict
from util.plotting import *
from src.sampling.Sampling import sampling_sndl


## Single Constructions

def ER_single(ntwk, save_path='data/ER_{ntwk}.txt'):
    save_path = save_path.format(ntwk=ntwk)
    G = nx.Graph()
    path = "data/" + str(ntwk) + '.txt'
    edgelist = list(np.genfromtxt(path, delimiter=",", dtype=str))
    for e in edgelist:
        G.add_edge(e[0], e[1])

    n = len(G.nodes())
    p = nx.density(G)

    G_er = nx.erdos_renyi_graph(n=n,p=p)
    nx.write_edgelist(G_er, save_path, data=False)


def WS_single(ntwk, p=0.1, random_orientation=False, save_path='data/WS_{ntwk}.txt'):
    # Watts-Strogatz model with baseline graph G and edge rewiring probability p 
    # G is undirected. Flip fair coins for each edge of G to get initial orientation.
    # For each oriented edge, resample the head node uniformly at random with probability p, independently. 
    # Do nothing for that edge with probability 1-p. 
    
    save_path = save_path.format(ntwk=ntwk)

    G = nx.Graph()
    path = "data/" + str(ntwk) + '.txt'
    edgelist = list(np.genfromtxt(path, delimiter=",", dtype=str))
    for e in edgelist:
        G.add_edge(e[0], e[1])

    # Give random orientation by crea
    if random_orientation: 
        G1 = random_orientation(G)
    else: #G is already a digraph 
        G1 = G

    nodes = list(G1.nodes())
    G_ws = nx.Graph()
    
    for e in G1.edges():
        U = np.random.rand()
        if U < p: 
            i = np.random.choice(np.arange(len(nodes)))
            v = nodes[i]
            G_ws.add_edge(e[0],v)
        else: 
            G_ws.add_edge(e[0],e[1])

    nx.write_edgelist(G_ws, save_path, data=False)


def BA_single(ntwk, m0=1, m=1, n=100, alpha=1, save_path='data/BA_{ntwk}.txt'):
    # Barabasi-Albert model with baseline graph G = single node with m0 self-loops 
    # Each new node has m edges pointing to some nodes in the existing graph 
    # alpha=1 -> preferential attachment: The head of each new directed edge is chosen randomly with probability 
    # proportional to the degree
    # alpha=0 ->: Uniform attachment: The head of each new directed edge is chosen uniformly at random
    # alpha \notin\{0,1} -> nonlinear preferential attachment: The head of each new directed edge is chosen 
    # randomly with probability proportional to the degree^alpha
    
    save_path = save_path.format(ntwk=ntwk)

    G0 = nx.Graph()
    path = "data/" + str(ntwk) + '.txt'
    edgelist = list(np.genfromtxt(path, delimiter=",", dtype=str))
    for e in edgelist:
        G0.add_edge(e[0], e[1])

    if G0 is not None: 
        G_ba = G0
    else: 
        G_ba = nx.MultiGraph() # baseline graph with a single node and m0 self-loops 
        for i in np.arange(m0):
            G_ba.add_edge(1,1)
        
    for s in np.arange(1,n):
        for j in np.arange(m):
            # form a degree distribution 
            degrees = np.asarray([G_ba.degree(n)**(alpha) for n in G_ba.nodes()])
            deg_dist = degrees*(1/np.sum(degrees))
            v = np.random.choice(G_ba.nodes(), p=deg_dist)
            G_ba.add_edge(s,v)

    nx.write_edgelist(G_ba, save_path, data=False)



def CM_single(ntwk, save_path='data/CM_{ntwk}.txt'):
    # Configuration model with degree sequence d = [d1, ... , dn] (a list or array)
    # di \ge 0 and sum to even 
    
    save_path = save_path.format(ntwk=ntwk)

    G0 = nx.Graph()
    path = "data/" + str(ntwk) + '.txt'
    edgelist = list(np.genfromtxt(path, delimiter=",", dtype=str))
    for e in edgelist:
        G0.add_edge(e[0], e[1])

    degrees = [G0.degree(v) for v in G0.nodes()]

    d = list(degrees)
    stubs_list = []
    for i in np.arange(len(d)):
        for j in np.arange(d[i]):
            stubs_list.append([i,j])

    G_cm = nx.MultiGraph()
    while len(stubs_list)>0:
        ss = np.random.choice(np.asarray(len(stubs_list)), 2, replace=False)
        s1 = ss[0]
        s2 = ss[1]
        half_edge1 = stubs_list[s1]
        half_edge2 = stubs_list[s2]
        G_cm.add_edge(half_edge1[0], half_edge2[0])
        stubs_list.remove(half_edge1)
        if s1 != s2:
            stubs_list.remove(half_edge2)
                    
    nx.write_edgelist(G_cm, save_path, data=False)

## Combined Constructions

def ER_combined(ntwk_list, save_path='data/ER_combined.txt'):
    G_combined = nx.Graph()

    # Calculate the average density across all networks
    total_edges = 0
    total_nodes = 0

    for ntwk in ntwk_list:
        path = "data/" + str(ntwk) + '.txt'
        edgelist = list(np.genfromtxt(path, delimiter=",", dtype=str))
        G = nx.Graph()
        for e in edgelist:
            G.add_edge(e[0], e[1])

        total_edges += G.number_of_edges()
        total_nodes += G.number_of_nodes()

    avg_density = total_edges / (total_nodes * (total_nodes - 1) / 2)

    # Generate ER graph with combined characteristics
    G_er = nx.erdos_renyi_graph(n=total_nodes, p=avg_density)
    nx.write_edgelist(G_er, save_path, data=False)



def WS_combined(ntwk_list, p=0.1, save_path='data/WS_combined.txt'):
    G_combined = nx.Graph()
    total_nodes = 0
    total_degree = 0

    # Calculate the average degree and total nodes across all networks
    for ntwk in ntwk_list:
        path = "data/" + str(ntwk) + '.txt'
        edgelist = list(np.genfromtxt(path, delimiter=",", dtype=str))
        G = nx.Graph()
        for e in edgelist:
            G.add_edge(e[0], e[1])

        total_nodes += G.number_of_nodes()
        total_degree += sum(dict(G.degree()).values())

    avg_degree = total_degree / total_nodes
    k = int(avg_degree / 2)  # WS model uses k as the number of neighbors

    # Generate WS graph with combined characteristics
    G_ws = nx.watts_strogatz_graph(n=total_nodes, k=k, p=p)
    nx.write_edgelist(G_ws, save_path, data=False)



def BA_combined(ntwk_list, m0=1, alpha=1, save_path='data/BA_combined.txt'):
    total_nodes = 0
    total_degree = 0

    # Calculate the average degree and total nodes across all networks
    for ntwk in ntwk_list:
        path = "data/" + str(ntwk) + '.txt'
        edgelist = list(np.genfromtxt(path, delimiter=",", dtype=str))
        G = nx.Graph()
        for e in edgelist:
            G.add_edge(e[0], e[1])

        total_nodes += G.number_of_nodes()
        total_degree += sum(dict(G.degree()).values())

    avg_degree = total_degree / total_nodes
    m = int(avg_degree / 2)  # BA model uses m as the number of edges to attach from a new node

    # Generate BA graph with combined characteristics
    G_ba = nx.barabasi_albert_graph(n=total_nodes, m=m)
    nx.write_edgelist(G_ba, save_path, data=False)



def CM_combined(ntwk_list, save_path='data/CM_combined.txt'):
    total_degree_sequence = []

    # Combine the degree sequences from all networks
    for ntwk in ntwk_list:
        path = "data/" + str(ntwk) + '.txt'
        edgelist = list(np.genfromtxt(path, delimiter=",", dtype=str))
        G = nx.Graph()
        for e in edgelist:
            G.add_edge(e[0], e[1])

        total_degree_sequence += [G.degree(v) for v in G.nodes()]

    # Generate CM graph with combined characteristics
    G_cm = nx.configuration_model(total_degree_sequence)
    G_cm = nx.Graph(G_cm)  # Remove parallel edges
    G_cm.remove_edges_from(nx.selfloop_edges(G_cm))
    nx.write_edgelist(G_cm, save_path, data=False)