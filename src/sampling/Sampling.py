# Author: Yi Wei
import numpy as np 
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
    if len(list_graphs) > 2:
        y_list = np.concatenate(y_list,axis=1)
    else:
        y_list = np.asarray(y_list).flatten()

    return X_list, y_list



def sampling_graph_classification(dataset, 
                                      sample_size=100, 
                                      k = 10, 
                                      has_node_feature = False,
                                      has_edge_feature = False,
                                      sampling_alg = 'pivot', #RW 
                                      skip_folded_hom=True):
    '''
    return: X.shape = [sample_size * num_Data_in_dataset, k*k]
            y.shape = [sample_size * num_Data_in_dataset, num_labels - 1]
    '''
    X_list = []
    y_list = []
    for idx, graph in enumerate(dataset):
        G = nw()
        edges =  graph.edge_index.T.tolist()
        G.add_edges(edges)

        # print(f"!!! edge_index: {edge_index}")
        # np.savetxt("edge_index.txt", edge_index, fmt='%d')

        X, emb = G.get_patches(k=k, sample_size=sample_size, 
                               sampling_alg=sampling_alg, 
                               skip_folded_hom=skip_folded_hom,
                               info_print=False)
        emb = np.asarray(emb).astype(int)
        # np.savetxt("emb.txt", emb, fmt='%d')
        # np.savetxt("X.txt", X, fmt='%d')

        real_sample_size = X.shape[1]
        if dataset.num_classes == 2:
            if graph.y.item() == 0:
                y = 0
            else:
                y = 1
            y_matrix = np.tile(y, (real_sample_size, 1)).T
            y_list.append(y_matrix)
        else:
            y = np.zeros(dataset.num_classes)
            if not graph.y.item() == 0:
                y[graph.y.item()-1] = 1
            y_matrix = np.tile(y, (real_sample_size, 1)).T
            y_list.append(y_matrix)
        
        
        if has_edge_feature:
            edge_features = np.zeros(shape=(X.shape[1], k*k*graph.edge_attr.shape[1]))
            for l in range(X.shape[1]):
                subgraph = X[:, l].reshape(k,k)
                edge_feature = np.zeros(shape=(k,k,graph.edge_attr.shape[1]))
                for i in range(k-1):
                    for j in range(i+1, k):
                        if subgraph[i,j] == 1:
                            ### find the corresponding nodes index by embedding
                            edge = np.array([emb[l][i], emb[l][j]])

                            row_index = np.nonzero(np.all(edges == edge, axis=1))[0][0]

                            feature = graph.edge_attr[row_index]
                            # print(feature)
                            # print(f"!!! {l}")
                            # print(f"!!! The shape of edge_feature: {edge_feature.shape}")
                            edge_feature[i,j] = feature
                            edge_feature[j,i] = feature
                edge_feature = edge_feature.reshape((1,-1))
                edge_features[l] = edge_feature
            X = np.vstack((X, edge_features.T))

        if has_node_feature:
            node_features = np.zeros(shape=(X.shape[1], k*graph.x.shape[1]))
            for l in range(X.shape[1]):
                node_feature = np.zeros((k,graph.x.shape[1]))
                for i in range(k):
                    node_feature[i] = graph.x[emb[l][i]]
                node_feature = node_feature.reshape((1,-1))
                node_features[l] = node_feature
            X = np.vstack((X, node_features.T))

        X_list.append(X)
        
    X_list = np.concatenate(X_list, axis=1)
    if dataset.num_classes == 2:
        y_list = np.asarray(y_list).flatten()
    else:
        y_list = np.concatenate(y_list,axis=1)
    
    return X_list, y_list