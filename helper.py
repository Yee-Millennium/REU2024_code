
import SMF_BCD as SMF
import numpy as np
from plotting import *
from run_SNLD import live_sample
import itertools
import os

def similarity(network1, network2, k = 30, xi = 5, n_components = 16):
    Output = live_sample(network1, network2, k = k, skip_folded_hom = False)
    X = Output[0]
    Y = Output[1][:, np.newaxis]
    Xtest, Ytest = X, Y
    SMF_Train = SMF.SDL_BCD([X, Y.T], X_test=[Xtest, Ytest.T], xi=xi, n_components=n_components)
    results_dict_new = SMF_Train.fit(iter=250, subsample_size=None, option = "filter",# search_radius_const=200*np.linalg.norm(X),
                                if_compute_recons_error=True, if_validate=True)
    W = results_dict_new.get('loading')[0]
    beta = results_dict_new.get('loading')[1]
    prediction1 = live_sample(network1, network1, k = k)[0]
    pred = []
    y = []
    for j in range(prediction1.shape[0]):
        a = beta[:,1:] @ W.T @ prediction1[:,j] + beta[:,0]
        pred.append(1/(1+np.exp(-a)[0]))
        y.append(0)
    prediction2 = live_sample(network2, network2, k = k)[0]
    for j in range(prediction2.shape[0]):
        a = beta[:,1:] @ W.T @ prediction2[:,j] + beta[:,0]
        pred.append(1/(1+np.exp(-a)[0]))
        y.append(1)
    return pred,y

# Make commatable with multinomial classification

def accuracy(network1, network2, k = 30, xi = 5, n_components = 16):
    Output = live_sample(network1, network2, k = k)
    X = Output[0]
    Y = Output[1][:, np.newaxis]
    Xtest, Ytest = X, Y
    SMF_Train = SMF.SDL_BCD([X, Y.T], X_test=[Xtest, Ytest.T], xi=xi, n_components=n_components)
    results_dict_new = SMF_Train.fit(iter=250, subsample_size=None, option = "filter",# search_radius_const=200*np.linalg.norm(X),
                                if_compute_recons_error=True, if_validate=True)
    W = results_dict_new.get('loading')[0]
    beta = results_dict_new.get('loading')[1]
    

# accuracy("MIT8", "Harvard1")

# similarity("MIT8", "Harvard1", "MIT8")
# # similarity("MIT8", "Harvard1", "Harvard1")
# similarity("Caltech36", "UCLA26", "Harvard1", k = 5)

