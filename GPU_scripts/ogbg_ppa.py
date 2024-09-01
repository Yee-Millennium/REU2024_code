from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.autograd import Variable


import sys
sys.path.append('../src/sampling')
from Sampling import sampling_graph_classification
sys.path.append('../src/supervised_NDL')
from SMF_torch import smf
sys.path.append('../src/util/plotting')
from plotting import *
import pickle


dataset_ppa = PygGraphPropPredDataset(name = 'ogbg-ppa')

### Before doing experiments, modify k(hidden_size), sample_size, and num_epochs
k = 20
sample_size=5

X , y = sampling_graph_classification(dataset = dataset_ppa, k = k, sample_size=sample_size, # 1, 5
                                          has_edge_feature=True, skip_folded_hom=True,
                                          info_print=False)

Accuracy = []
test_size = 0.3

X_train, X_test, Y_train, Y_test = train_test_split(X.T, y.T, test_size=test_size, random_state=2)
# print(X_train.shape)
# print(Y_train.shape)

X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(Y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(Y_test)).long()
# print(f"y_train's ndim: {y_train.ndim}")
# print(f"y_test: {y_test.shape}")

smf_model = smf(X_train, y_train, hidden_size=k, device='cuda')
results_dict = smf_model.fit(num_epochs=200,
               lr_classification=0.01,
               lr_matrix_factorization=0.01,
               xi=5,
               initialize='spectral',
               W_nonnegativity=True,
               H_nonnegativity=True,
               test_data=[X_test, y_test],
               record_recons_error=False)

W = results_dict.get('loading')[0]
beta= results_dict.get('loading')[1]
H = results_dict.get('code')

display_dict_and_graph(save_path=
                       f'./ogbg_ppa', W=W[:k**2], regression_coeff=beta.T, 
                       fig_size=[15,15], plot_graph_only=True)

### Save the result 
with open(f"./results_dict_{k}_{sample_size}.pkl", 'wb') as file:
    pickle.dump(results_dict, file)