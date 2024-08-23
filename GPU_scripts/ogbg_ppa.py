from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.autograd import Variable
from src.supervised_NDL.SMF_BCD import SDL_BCD
from src.supervised_NDL.SMF_torch import smf
import sys

sys.path.append("../src")
from src.sampling.Sampling import sampling_graph_classification


dataset_ppa = PygGraphPropPredDataset(name = 'ogbg-ppa')

### Before doing experiments, modify k(hidden_size) and sample_size.
k = 1

X , y = sampling_graph_classification(dataset = dataset_ppa, k = k, sample_size=1, 
                                          has_edge_feature=True)

Accuracy = []
test_size = 0.3

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=2)
# print(X_train.shape)
# print(Y_train.shape)

X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(Y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(Y_test)).long()
# print(f"y_train's ndim: {y_train.ndim}")
# print(f"y_test: {y_test.shape}")

smf_model = smf(X_train, y_train, hidden_size=k, device='cuda')
smf_model.fit(num_epochs=500,
               lr_classification=0.01,
               lr_matrix_factorization=0.01,
               xi=1,
               initialize='spectral',
               W_nonnegativity=True,
               H_nonnegativity=True,
               test_data=[X_test, y_test],
               record_recons_error=True)