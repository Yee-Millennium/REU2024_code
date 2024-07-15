import numpy as np 
from NNetwork import NNetwork as nn


### Maybe todo: larger network deserves more samples 



### Return: a matrix of size k^2 * (len(list_graphs)*sample_size),
### and a vector of size len(list_graphs)*sample_size
def sampling(list_graphs: list, labels = None, k = 20, sample_size = 100
             , sampling_alg = 'pivot' #RW
             ,skip_folded_hom=True
             ):
    # list of graphs in NNetwork format
    len_networks = len(list_graphs)
    X_list = []
    embs_list = []
    i = 0

    # Construct the matrix
    for G in list_graphs:
        if sampling_alg != 'RW':
            X, embs = G.get_patches(k=k, sample_size=sample_size,
                                    sampling_alg = sampling_alg,
                                    skip_folded_hom=skip_folded_hom)
            X_list.append(X)
            embs_list.append(embs)
        else:
            X = []
            embs = []
            for i in np.arange(sample_size):
                H = G.k_node_ind_subgraph(k=k)
                while H is None:
                    H = G.k_node_ind_subgraph(k=k)
                Adj = H.get_adjacency_matrix()
                X.append(Adj.reshape(1,-1))
                embs.append(list(H.nodes()))
            X = np.asarray(X)
            X = X[:,0,:].T

            X_list.append(X)
            embs_list.append(embs)

    X_list = np.asarray(X_list)
    X_list = np.hstack(X_list)

    # Construct the labels
    if labels == None:
        labels = np.arange(0, len_networks, 1)

    label_vec = []
    for i in range(len_networks):
        label_vec.append(np.full(sample_size, labels[i]))
    
    label_vec = np.asarray(label_vec)
    label_vec = label_vec.reshape(-1)

    return X_list, label_vec