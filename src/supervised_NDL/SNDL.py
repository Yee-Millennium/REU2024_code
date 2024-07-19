import numpy as np
from NNetwork import NNetwork as nn
from src.sampling.Sampling import sampling_SNLD
from . import SMF_BCD


def sndl_equalEdge(graph_list, sample_size_1, sample_size_2, k, xi, n_components=16, iter = 250):
    '''
    1. Given input graph_list, sample_size_1, k, 
    sample sample_size_1 number of k-subgraphs of the first network, and 
    sample sample_size_2 number of k-subgraphs of the second network.
    Notice that we sample different number of subgraphs for different networks to make
    edges from these two respectively are equal.
    2. Given xi, n_components and the samples with subgraphs and features,
    use SMF_BCD algorithms to solve the corresponding supervised matrix factorization 
    problem, and get the result W (dictionary), beta (coefficients) and H (code).
    '''

    # sample_size_2 = sample_size_1 * len(graph_list[1].get_edges()) / len (graph_list[0].get_edges())
    # sample_size_2 = int(sample_size_2)
    print(f"This is the second sample_size: {sample_size_2}")
    X, y = sampling_SNLD(graph_list, k=k, sample_size_list=[sample_size_1, sample_size_2])
    # np.sum(X[:, :sample_size_1])
    print(np.sum(X[:, :sample_size_1]))
    print(np.sum(X[:, sample_size_1:]))
    y = y.reshape(-1,1)

    # SMF_W solve the SNDL
    SMF_Train = SMF_BCD.SDL_BCD([X, y.T], X_test=[X, y.T], xi= xi, n_components=n_components)
    results_dict = SMF_Train.fit(iter=iter, subsample_size=None, option = "filter",# search_radius_const=200*np.linalg.norm(X),
                                if_compute_recons_error=True, if_validate=True)
    
    W = results_dict.get('loading')[0]
    beta= results_dict.get('loading')[1]
    H = results_dict.get('code')

    return W, beta, H



def sndl_reg(G3, W, beta, n3):
    '''
    Given W, beta, which we obtain from solving a SMF problem of two networks, 
    and G3, which is a network different with the previous two networks,
    sample given n3 number of subgraphs, 
    and predict the probability by logistics of each subgraph to be 1.
    Then return the averaged probability.
    '''
    k = int(len(W)**0.5)
    X, embs = G3.get_patches(k=k, sample_size=n3)

    p_sum = 0
    for i in range(n3):
        a = beta[:,1:] @ W.T @ X[:,i] + beta[:,0]
        p = 1/(1+np.exp(-a))
        p_sum += p
    
    prob = p_sum / n3

    return prob

