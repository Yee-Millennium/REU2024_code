import numpy as np 
from NNetwork import NNetwork as nn

### Return: a matrix of size k^2 * (len(list_graphs)*sample_size),
### and a vector of size len(list_graphs)*sample_size
def sampling_SNLD(list_graphs: list, labels = None, k = 20, sample_size_list = None
             , sample_size = 200,
             sampling_alg = 'pivot' #RW
             ,skip_folded_hom=True
             ):
    # list of graphs in NNetwork format
    len_networks = len(list_graphs)
    X_list = []
    embs_list = []
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

    if len_networks == 2:
        label_vec = np.zeros((1,sum(sample_size_list)))
        label_vec[0, sample_size_list[0]:] = 1
    else:
        label_vec = np.zeros((len_networks, sum(sample_size_list)))
        for i in range(len_networks):
            label_vec[i, sum(sample_size_list[:i]) : sum(sample_size_list[:(i+1)])] = 1

    return X_list, label_vec


def live_sample(graph1, graph2, k=30, tolerance=0.05):
    ntwk_list = [graph1, graph2]
    graph_list = []
    for ntwk in ntwk_list:
        path = "data/" + str(ntwk) + '.txt'
        G = nn.NNetwork()
        G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
        graph_list.append(G)
    if graph1 == graph2:
        return sampling_SNLD(graph_list, k=k, sample_size_list =[guess1,guess2]), 500, 500
    
    guess1, guess2 = 500, 500
    A, y = sampling_SNLD(graph_list, k=k, sample_size_list=[guess1, guess2])
    graph1_val = np.sum(A[:, :guess1])
    graph2_val = np.sum(A[:, guess1:])
    if graph1_val > graph2_val:
        difference = graph1_val - graph2_val
        graph2_unit = graph2_val/500
        guess2 += difference//graph2_unit
    else:
        difference = graph2_val - graph1_val
        graph1_unit = graph1_val/500
        guess1 += difference//graph1_unit
    A, y = sampling_SNLD(graph_list, k=k, sample_size_list=[int(guess1), int(guess2)])
    return A, y, guess1, guess2