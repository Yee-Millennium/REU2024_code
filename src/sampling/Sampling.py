import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from NNetwork_modified.NNetwork import NNetwork as nw


def sampling_sndl(list_graphs: list, 
                  k = 20, 
                  sample_size_list = None, 
                  sample_size = 200,
                  sampling_alg = 'pivot', #RW
                  skip_folded_hom=True
             ):
    '''
    args:
        list_graphs: list of different graphs for sampling
        k: num of nodes in each sampled subgraph
        sample_size_list: sample sizes for different graphs in list_graphs
        sample_size: base sample size when sample_size_list is none
        sampling_alg: 'pivot'/'RW'
        skip_folded_hom: reject overlapping sampled k-path

    returns:
        X_list: numpy array, [num of nodes * num of nodes(subgraph), num of samples]; horizontally concatenated adjacent matrices after vectorization of 
                sampled subgraphs for supervised network dictionary learning
        y_list: numpy array, [num of labels-1, num of samples]; matrix of labels (without 0) for supervised network dictionary learning
    '''
    # list of graphs in NNetwork format
    len_networks = len(list_graphs)
    X_list = []
    y_list = []
    i = 0

    if sample_size_list is None:
        sample_size_list = [sample_size] * len_networks  # Use default sample_size for each graph

    # Construct the matrix
    for idx, G in enumerate(list_graphs):
        sample_size = sample_size_list[idx]
        if sampling_alg != 'RW':
            X, embs = G.get_patches(k=k, sample_size=sample_size,
                                    sampling_alg = sampling_alg,
                                    skip_folded_hom=skip_folded_hom)
            X_list.append(X)
            
        else:
            X = []
            embs = []
            for i in np.arange(sample_size):
                H = G.k_node_ind_subgraph(k=k)
                while H is None:
                    H = G.k_node_ind_subgraph(k=k)
                Adj = H.get_adjacency_matrix()
                X.append(Adj.reshape(1,-1))
            X = np.asarray(X)
            X = X[:,0,:].T
            X_list.append(X)
        
        # construct matrix of labels
        real_sample_size = X.shape[1]
        y = np.zeros(len_networks-1)
        if idx != 0:
            y[idx-1] = 1 
        y_matrix = np.tile(y, (real_sample_size, 1)).T
        y_list.append(y_matrix)
    
    X_list = np.concatenate(X_list, axis=1)
    y_list = np.concatenate(y_list,axis=1)

    return X_list, y_list