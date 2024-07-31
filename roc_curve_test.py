import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from helper import live_sample
import SMF_BCD as SMF
import json
import matplotlib.pyplot as plt
import os


# network1 = "bn-fly-drosophila_medulla_1"
network2 = "bn-mouse-kasthuri_graph_v4"

network1 = "MIT8"
# network2 = "UCLA26"

def run(network1, network2, k = 20):
    # pred, y, W, beta, threshold = similarity(network1, network2, k=k)
    # fpr, tpr, thresholds = roc_curve(y, pred)
    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_idx]
    # test_predictions_network1 = predict_with_threshold(network1, k, W, beta, optimal_threshold)
    # test_predictions_network2 = predict_with_threshold(network2, k, W, beta, optimal_threshold)
    # return 1-sum(test_predictions_network1)/len(test_predictions_network1), (sum(test_predictions_network2)/len(test_predictions_network2)), optimal_threshold
    return similarity(network1, network2, k=k)

def similarity(network1, network2, k=30, xi=5, n_components=16):
    Output = live_sample(network1, network2, k=k)
    X = Output[0]
    Y = Output[1][:, np.newaxis]
    Xtrain, Ytrain = X, Y

    json_file_path = "output.json"
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as json_file:
            try:
                data_to_write = json.load(json_file)
            except json.JSONDecodeError:
                data_to_write = {}
    else:
        data_to_write = {}
        
    key = f"{network1}-{network2}-{k}-{xi}-{n_components}"
    # if key in data_to_write:
    if False:
        W = np.array(data_to_write[key]["W"])
        beta = np.array(data_to_write[key]["beta"])
    else:    
        SMF_Train = SMF.SDL_BCD([X, Y.T], X_test=[Xtrain, Ytrain.T], xi=xi, n_components=n_components)
        results_dict_new = SMF_Train.fit(iter=250, subsample_size=None, option="filter", if_compute_recons_error=True, if_validate=True)
        
        W = results_dict_new.get('loading')[0]
        beta = results_dict_new.get('loading')[1]
        threshold = results_dict_new.get('Opt_threshold')
        W_list = W.tolist()
        beta_list = beta.tolist()

        new_data = {
            key: {
                "W": W_list,
                "beta": beta_list
            }
        }
        data_to_write.update(new_data)
        with open(json_file_path, "w") as json_file:
            json.dump(data_to_write, json_file, indent=4)
    
    # fpr, tpr, thresholds = roc_curve(y, pred)
    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_idx]
    
    # Testing
    prediction1 = live_sample(network1, network1, k=k)[0]
    pred = []
    y = []
    correct = 0
    for j in range(prediction1.shape[0]):
        a = beta[:, 1:] @ W.T @ prediction1[:, j] + beta[:, 0]
        if a < threshold:
            correct += 1
            pred.append(0)
        else:
            pred.append(1)
        # pred.append(1 / (1 + np.exp(-a)[0]))
        y.append(0)
    
    prediction2 = live_sample(network2, network2, k=k)[0]
    
    for j in range(prediction2.shape[0]):
        a = beta[:, 1:] @ W.T @ prediction2[:, j] + beta[:, 0]
        if a < threshold:
            pred.append(0)
        else:
            pred.append(1)
            correct += 1
        # pred.append(1 / (1 + np.exp(-a)[0]))
        y.append(1)
    
    
    # return np.array(pred), np.array(y), W, beta, threshold
    return correct/len(y), threshold

def plot_k():
    accu = []
    for i in range(5,10):
        print(i)
        accu.append(run(i))
    for i in range(10,15):
        print(i)
        accu.append(run(i))
    for i in range(15,20):
        print(i)
        accu.append(run(i))
    for i in range(20,25):
        print(i)
        accu.append(run(i))
    for i in range(25,30):
        print(i)
        accu.append(run(i))
    accu1 = [(i[0]) for i in accu]
    accu2 = [(i[1]) for i in accu]
    thresholds = [i[2] for i in accu]
    x_values = np.arange(5,29)
    plt.plot(x_values, accu1, label='Accuracy 1')
    plt.plot(x_values, accu2, label='Accuracy 2')
    plt.plot(x_values, thresholds, label='Thresholds')

    plt.legend()
    plt.show()

# print(run(15))

for i in range(35, 50):
    print(f"$$$$$${run(network1, network2, i)}$$$$$")