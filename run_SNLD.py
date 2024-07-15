import numpy as np 
from NNetwork import NNetwork as nn
from src.sampling.sampling import sampling

def run_SNLD():
    #ntwk_list = ['Caltech36', 'UCLA26', 'MIT8', 'Harvard1']
    #ntwk_list = ['Caltech36', 'UCLA26']
    ntwk_list = ['MIT8', 'Harvard1']
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
    A, y = sampling(graph_list, k = 30, sample_size_list=[1000, 1000])
    print(A,y)
    # Check
    np.savetxt('Output/At_mh.txt', A, fmt='%d')
    np.savetxt('Output/yt_mh.txt', y, fmt='%d')

    np.savetxt('Output/A_mh.txt', A, fmt='%d')
    np.savetxt('Output/y_mh.txt', y, fmt='%d')

    print(f"\nThis is the shape of A_mh: {A.shape}")
    print(f"\nThis is the shape of y_mh: {y.shape}")


if  __name__ == '__main__':
    run_SNLD()
