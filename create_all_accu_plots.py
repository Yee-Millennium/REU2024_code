import roc_curve_test as roc
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


NETWORK_LIST = ['Caltech36', 'UCLA26', 'MIT8', 'Harvard1']

network_pairs = list(combinations(NETWORK_LIST, 2))

lower_bound = 5
upper_bound = 30
for i in network_pairs:
    accu = []
    for j in range(lower_bound, upper_bound):
        accu.append(roc.run(network1 = i[0], network2 = i[1], k = j))
    accu1 = [i[0] for i in accu]
    thresholds = [i[1] for i in accu]
    x_values = np.arange(lower_bound, upper_bound)
    plt.plot(x_values, accu1, label='Accuracy 1')
    plt.plot(x_values, thresholds, label='Thresholds')
    plt.legend()
    
    plt.savefig(f"Output/accu_plots/{i[0]}{i[1]}.png")
    plt.clf()
        