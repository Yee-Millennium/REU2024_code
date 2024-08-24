# REU2024_code
 This repository contains codes of REU 2024 with PI Prof. Hanbaek Lyu.

 ## File Description

 1. ***sampling.ipynb***: Designed for sampling subgraphs for SNDL problem. You will get a matrix consists of subgraphs after vectorization, and a feature vector. If you want to sample subgraphs to just get an adjancy matrix, refer to NNetwork package.
 2. ***NDL_tutorial.ipynb***: Tutorial of NDL methods, mainly including code demonstration of learning latent motifs, and network reconstruction.
 3. ***SMF_tutorial.ipynb***: Code demonstration of SMF algorithm.
 4. ***SNDL_tutorial.ipynb***: Code demonstration of process to solve a supervised network dictionary learning problem, to display the dictionary, and to predict similarity among networks.
 5. ***MNIST_test.ipynb***: Code test for multiclass functionality of BCD algorithm on MNIST dataset
 6. ***Binary_Prediction_Plot.ipynb***: Visualization function and example usage for binary case. For the input network list, display the prediction score (in the format of heatmap) between each unordered pair and distinct networks.
 7. ***Multiclass_Dictionary_Plot.ipynb***: Visualization function and example usage for multiclass case. For the input network list and assigned baseline (3 networks), display the prediction score on a 3-dimensional triangle plot, with three standard vertices.
 8. ***Visualization_Experiment.ipynb***: Functions of constructing synthetic networks based on specific given network. Experiment with synthetic networks and biological networks (BioGRID PPI) using functions in `visualization.py`.
