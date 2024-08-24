# Author: Yi Wei
# Description: Sampling and training for supervised network dictionary learning tasks

import numpy as np
import sys
import os
from NNetwork import NNetwork as nn
sys.path.insert(0, r"C:\Users\KuangQi\Desktop\REU_main\REU2024_code")
from src.sampling.Sampling import sampling_sndl
from src.supervised_NDL import SMF_BCD


def sndl_equalEdge(graph_list, 
                   sample_size_list = None,
                   k = 16, 
                   xi = 5, 
                   threshold = 0.5,
                   n_components=16, 
                   iter = 250, 
                   base_sample_size = 500,
                   subsample_number = 300,
                   if_compute_recons_error=False, 
                   if_validate=False,
                   skip_folded_hom=True):
    '''
    args:
        graph_list: graphs which we do supervised network dictionary learning task on
        sample_size_list: the number of subgraphs you sample from each graph from graph_list
        k: nodes number in each subgraph
        xi: trade-off parameter in SNDL
        n_components: the number of subgraphs in the learned dictionary
        iter: iteration number for SNDL
        base_sample_size: the sample size for the first graph in the graph_list when you want
                            to have balanced edge numbers of different graphs
        subsample_number: the sample size which we utilize to find a balanced sample_size_list

    return:
        W: learned dictionary (ref. paper 'Supervised Matrix Factorization: Local Landscape Analysis and Applications')
        beta: learned coefficients
        H: learned code with respect to the learned dictionary W and the sample dataset X
    
    1. If sample_size_list is not given, find a balanced sample_size_list which makes the total number of 
    edges in the dataset matrix of different graphs are similar
    2. Use SMF_BCD algorithms to solve the corresponding supervised matrix factorization 
    problem, and get the result W (dictionary), beta (coefficients) and H (code).
    '''
    ### Sample subgraphs to ensure similar edge number in each subgraph
    if sample_size_list == None:
        temp_X, temp_y = sampling_sndl(graph_list, sample_size_list=[subsample_number]*len(graph_list)
                                       , k=k, skip_folded_hom=skip_folded_hom)
        edge_num = []
        ratio = []
        size_list = [base_sample_size]
        for i in range(len(graph_list)):
            edge_num.append(np.sum(temp_X[:, i*subsample_number:(i+1)*subsample_number]))
            if i >= 1:
                ratio.append(edge_num[0] / edge_num[i])
                size_list.append(round(base_sample_size * ratio[i-1]))
        print(f" !!! The balanced size_list: {size_list}")
    else:
        size_list = sample_size_list

    X, y = sampling_sndl(graph_list, k=k, sample_size_list=size_list, skip_folded_hom=skip_folded_hom)
    # SMF_W solve the SNDL
    SMF_Train = SMF_BCD.SDL_BCD([X, y], X_test=[X, y], xi= xi, n_components=n_components)
    results_dict = SMF_Train.fit(iter=iter, subsample_size=None,# search_radius_const=200*np.linalg.norm(X),
                                if_compute_recons_error=if_compute_recons_error, if_validate=
                                if_validate, threshold=threshold)
    
    W = results_dict.get('loading')[0]
    beta= results_dict.get('loading')[1]
    H = results_dict.get('code')

    return W, beta, H



def sndl_predict(G3, W, beta, n3):
    '''
    args:
        G3: network which we'll predict
        W: learned dictionary
        beta: learned coefficients and interceptions
        n3: number of sample subgraphs from G3, which we will use to predict

    return:
        prob: probabilities for G3 to be each label (Notice: No prediction for label 0)

    Given W, beta, which we obtain from solving a SMF problem of some networks, 
    and G3, which is a network different with the previous networks used in training,
    sample given n3 number of subgraphs, 
    and predict the probabilities by logistics of each subgraph to be each label.
    Then return the averaged probability.
    '''
    k = int(len(W)**0.5)
    X, embs = G3.get_patches(k=k, sample_size=n3)

    p_sum = 0
    for i in range(n3):
        normalizer = 1 + np.sum(np.exp(beta[:, 1:] @ W.T @ X[:, i] + beta[:, 0]))
        p = np.exp(beta[:, 1:] @ W.T @ X[:, i] + beta[:, 0]) / normalizer
        p_sum += p
    
    prob = p_sum / n3
    prob = np.insert(prob, 0, 1-np.sum(prob))

    return prob
