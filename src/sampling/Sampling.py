import numpy as np 
from NNetwork import NNetwork as nn

### Return: a matrix of size k^2 * (len(list_graphs)*sample_size),
### and a vector of size len(list_graphs)*sample_size
def sampling_sndl(list_graphs: list, 
                  labels = None, 
                  k = 20, 
                  sample_size_list = None, 
                  sample_size = 200,
                  sampling_alg = 'pivot', #RW
                  skip_folded_hom=True
             ):
    # list of graphs in NNetwork format
    len_networks = len(list_graphs)
    X_list = []
    embs_list = []
    i = 0

    if sample_size_list is None:
        sample_size_list = [sample_size] * len_networks  # Use default sample_size for each graph

    # Construct the matrix X
    for idx, G in enumerate(list_graphs):
        sample_size = sample_size_list[idx]
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
    
    X_list = np.concatenate(X_list, axis=1)


    # Construct the labels
    if labels == None:
        labels = np.arange(0, len_networks, 1)

    label_matrix = np.zeros((len_networks-1, sum(sample_size_list)))
    for i in range(len_networks-1):
        label_matrix[i, sum(sample_size_list[:i+1]) : sum(sample_size_list[:(i+2)])] = 1

    return X_list, label_matrix
