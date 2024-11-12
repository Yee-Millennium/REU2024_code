# from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
# from sklearn.model_selection import train_test_split
# import numpy as np
# import torch
# from torch.autograd import Variable
# import pickle


# import sys

# sys.path.append('../src/sampling')
# from Sampling import sampling_graph_classification

# sys.path.append('../src/supervised_NDL')
# from SMF_torch import smf

# sys.path.append('../util')
# from plotting import *


# dataset_ppa = PygGraphPropPredDataset(name = 'ogbg-ppa')

# ### Before doing experiments, modify k(hidden_size), sample_size, and num_epochs
# k = 20
# sample_size=5

# X , y = sampling_graph_classification(dataset = dataset_ppa, k = k, sample_size=sample_size, # 1, 5
#                                           has_edge_feature=True, skip_folded_hom=True,
#                                           info_print=False)

# Accuracy = []
# test_size = 0.3

# X_train, X_test, Y_train, Y_test = train_test_split(X.T, y.T, test_size=test_size, random_state=2)
# # print(X_train.shape)
# # print(Y_train.shape)

# X_train = Variable(torch.from_numpy(X_train)).float()
# y_train = Variable(torch.from_numpy(Y_train)).long()
# X_test = Variable(torch.from_numpy(X_test)).float()
# y_test = Variable(torch.from_numpy(Y_test)).long()
# # print(f"y_train's ndim: {y_train.ndim}")
# # print(f"y_test: {y_test.shape}")

# smf_model = smf(X_train, y_train, hidden_size=k, device='cuda')
# results_dict = smf_model.fit(num_epochs=200,
#                lr_classification=0.01,
#                lr_matrix_factorization=0.01,
#                xi=5,
#                initialize='spectral',
#                W_nonnegativity=True,
#                H_nonnegativity=True,
#                test_data=[X_test, y_test],
#                record_recons_error=False)

# W = results_dict.get('loading')[0]
# beta= results_dict.get('loading')[1]
# H = results_dict.get('code')

# display_dict_and_graph(save_path=
#                        f'./ogbg_ppa', W=W[:k**2], regression_coeff=beta.T, 
#                        fig_size=[15,15], plot_graph_only=True)

# ### Save the result 
# with open(f"./results_dict_{k}_{sample_size}.pkl", 'wb') as file:
#     pickle.dump(results_dict, file)

import multiprocessing
from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.autograd import Variable
import pickle
import sys

sys.path.append('../src/sampling')
from Sampling import sampling_graph_classification  # Import as is

sys.path.append('../src/supervised_NDL')
from SMF_torch import smf

sys.path.append('../util')
from plotting import *

def process_chunk(args):
    dataset_chunk, k, sample_size, has_edge_feature, skip_folded_hom, info_print = args
    # Call the sampling_graph_classification function on the chunk
    X_chunk, y_chunk = sampling_graph_classification(
        dataset=dataset_chunk,
        k=k,
        sample_size=sample_size,
        has_edge_feature=has_edge_feature,
        skip_folded_hom=skip_folded_hom,
        info_print=info_print
    )
    return X_chunk, y_chunk

def parallel_sampling_graph_classification(
    dataset,
    k,
    sample_size,
    has_edge_feature=True,
    skip_folded_hom=False,
    info_print=False,
    num_workers=1
):
    # Split the dataset into chunks
    num_graphs = len(dataset)
    chunk_size = num_graphs // num_workers
    dataset_chunks = [
        dataset[i * chunk_size:(i + 1) * chunk_size] if i < num_workers - 1 else dataset[i * chunk_size:]
        for i in range(num_workers)
    ]
    
    args_list = [
        (chunk, k, sample_size, has_edge_feature, skip_folded_hom, info_print)
        for chunk in dataset_chunks
    ]

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_chunk, args_list)

    # Combine the results
    X_list = [result[0] for result in results]
    y_list = [result[1] for result in results]
    X = np.hstack(X_list)
    y = np.hstack(y_list)
    return X, y

if __name__ == '__main__':
    # Set the number of workers to the number of available CPUs
    num_workers = multiprocessing.cpu_count()

    dataset_ppa = PygGraphPropPredDataset(name='ogbg-ppa')

    # Modify k (hidden_size) and sample_size as needed
    k = 20
    sample_size = 5

    # Use the parallel version of sampling_graph_classification
    X, y = parallel_sampling_graph_classification(
        dataset=dataset_ppa,
        k=k,
        sample_size=sample_size,
        has_edge_feature=True,
        skip_folded_hom=True,
        info_print=True,
        num_workers=num_workers  # Utilize multiple CPUs
    )

    Accuracy = []
    test_size = 0.3

    X_train, X_test, Y_train, Y_test = train_test_split(
        X.T, y.T, test_size=test_size, random_state=2
    )

    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(Y_train)).long()
    X_test = Variable(torch.from_numpy(X_test)).float()
    y_test = Variable(torch.from_numpy(Y_test)).long()

    smf_model = smf(X_train, y_train, hidden_size=k, device='cuda')
    results_dict = smf_model.fit(
        num_epochs=200,
        lr_classification=0.01,
        lr_matrix_factorization=0.01,
        xi=5,
        initialize='spectral',
        W_nonnegativity=True,
        H_nonnegativity=True,
        test_data=[X_test, y_test],
        record_recons_error=False
    )

    W = results_dict.get('loading')[0]
    beta = results_dict.get('loading')[1]
    H = results_dict.get('code')

    display_dict_and_graph(
        save_path=f'./ogbg_ppa',
        W=W[:k**2],
        regression_coeff=beta.T,
        fig_size=[15, 15],
        plot_graph_only=True
    )

    # Save the result
    with open(f"./results_dict_{k}_{sample_size}.pkl", 'wb') as file:
        pickle.dump(results_dict, file)