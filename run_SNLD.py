import numpy as np 
from NNetwork import NNetwork as nn
from src.sampling.sampling import sampling
import os
import sys

NETWORK_LIST = ['Caltech36', 'UCLA26', 'MIT8', 'Harvard1']

def save_samples(graph1, graph2):
    ntwk_list = [graph1, graph2]
    
    graph_list = []
    for ntwk in ntwk_list:
        path = "data/" + str(ntwk) + '.txt'
        G = nn.NNetwork()
        G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
        graph_list.append(G)
    # Do sampling, and get the matrix and the feature vector
    A, y = sampling(graph_list, k = 30, sample_size_list=[500, 2000])
    print(A,y)
    # Check
    # np.savetxt('Output/At.txt', A, fmt='%d')
    # np.savetxt('Output/yt.txt', y, fmt='%d')

    np.savetxt('Output/A2.txt', A, fmt='%d')
    np.savetxt('Output/y2.txt', y, fmt='%d')
    
    # np.savetxt('Output/At_mh.txt', A, fmt='%d')
    # np.savetxt('Output/yt_mh.txt', y, fmt='%d')

    # np.savetxt('Output/A_mh.txt', A, fmt='%d')
    # np.savetxt('Output/y_mh.txt', y, fmt='%d')

    print(f"\nThis is the shape of A_mh: {A.shape}")
    print(f"\nThis is the shape of y_mh: {y.shape}")

def live_sample(graph1, graph2, k=30, tolerance=0.05):
    ntwk_list = [graph1, graph2]
    graph_list = []
    for ntwk in ntwk_list:
        path = "data/" + str(ntwk) + '.txt'
        G = nn.NNetwork()
        G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
        graph_list.append(G)
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


def main():
    sys.stdout = open(os.devnull, 'w')
    # print(live_sample('Caltech36', 'UCLA26'))
    save_samples("Caltech36", "Caltech36")
    

if  __name__ == '__main__':
    main()
