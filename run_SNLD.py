import numpy as np 
from NNetwork import NNetwork as nn
from src.sampling.sampling import sampling


#ntwk_list = ['Caltech36', 'UCLA26', 'MIT8', 'Harvard1']
ntwk_list = ['Caltech36', 'UCLA26']
sampling_alg = 'pivot'
save_folder = 'output/'

graph_list = []
for ntwk in ntwk_list:
    ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
    path = "data/" + str(ntwk) + '.txt'
    G = nn.NNetwork()
    G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
    graph_list.append(G)



# Do sampling, and get the matrix and the feature vector
A, y = sampling(graph_list, k = 30, sample_size=100)
print(f"This is the shape of A: {A}")
print(f"\nThis is the shape of y: {y}")