import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from helper import live_sample
import SMF_BCD as SMF
import json

network1 = "Caltech36"
network2 = "UCLA26"

def run(k = 20):
    pred, y, W, beta = similarity(network1, network2, k=k)
    fpr, tpr, thresholds = roc_curve(y, pred)
    # auc = roc_auc_score(y, pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    test_predictions_network1 = predict_with_threshold(network1, k, W, beta, optimal_threshold)
    test_predictions_network2 = predict_with_threshold(network2, k, W, beta, optimal_threshold)
    return 1-sum(test_predictions_network1)/len(test_predictions_network1), (sum(test_predictions_network2)/len(test_predictions_network2)), optimal_threshold

def similarity(network1, network2, k=30, xi=5, n_components=16):
    Output = live_sample(network1, network2, k=k)
    X = Output[0]
    Y = Output[1][:, np.newaxis]
    Xtest, Ytest = X, Y
    
    SMF_Train = SMF.SDL_BCD([X, Y.T], X_test=[Xtest, Ytest.T], xi=xi, n_components=n_components)
    results_dict_new = SMF_Train.fit(iter=250, subsample_size=None, option="filter", if_compute_recons_error=True, if_validate=True)
    
    W = results_dict_new.get('loading')[0]
    beta = results_dict_new.get('loading')[1]
    W_list = W.tolist()
    beta_list = beta.tolist()

    key = f"{network1}-{network2}-{k}-{xi}-{n_components}"
    data_to_write = {
        key: {
            "W": W_list,
            "beta": beta_list
        }
    }

    with open("output.json", "w") as json_file:
        json.dump(data_to_write, json_file, indent=4)

    prediction1 = live_sample(network1, network1, k=k)[0]
    pred = []
    y = []
    
    for j in range(prediction1.shape[0]):
        a = beta[:, 1:] @ W.T @ prediction1[:, j] + beta[:, 0]
        pred.append(1 / (1 + np.exp(-a)[0]))
        y.append(0)
    
    prediction2 = live_sample(network2, network2, k=k)[0]
    
    for j in range(prediction2.shape[0]):
        a = beta[:, 1:] @ W.T @ prediction2[:, j] + beta[:, 0]
        pred.append(1 / (1 + np.exp(-a)[0]))
        y.append(1)
    
    return np.array(pred), np.array(y), W, beta

def predict_with_threshold(network, k, W, beta, threshold):
    prediction = live_sample(network, network, k=k)[0]
    predictions = []
    for j in range(prediction.shape[0]):
        a = beta[:, 1:] @ W.T @ prediction[:, j] + beta[:, 0]
        prob = 1 / (1 + np.exp(-a)[0])
        pred_label = 1 if prob >= threshold else 0
        predictions.append(pred_label)
    return np.array(predictions)


# print(run(20))