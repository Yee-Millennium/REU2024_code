# Author: Yi Wei

# from utils.onmf.onmf import Online_NMF
from ndl import Online_NMF
from NNetwork import NNetwork as nn
from ndl import utils
import numpy as np
import itertools
from time import time
from sklearn.decomposition import SparseCoder
import matplotlib.pyplot as plt
import networkx as nx
import os
import psutil
import matplotlib.gridspec as gridspec
import sys
import random
from tqdm import trange

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

DEBUG = False


class NetDictLearner_RW():
    def __init__(self,
                 G,
                 n_components=100,
                 MCMC_iterations=500,
                 sub_iterations=100,
                 sample_size=1000,
                 k=21,
                 alpha=None,
                 is_glauber_dict=True,
                 is_glauber_recons=True,
                 Pivot_exact_MH_rule=False,
                 ONMF_subsample=True,
                 batch_size=10,
                 if_wtd_network=False,
                 if_tensor_ntwk=False,
                 omit_folded_edges=False):

        """
        Constructor for the NetDictLearner Class

        Parameters
        ----------
        G: Wtd_NNetwork object
            Network to use for learning and reconstruction.

        n_components: int
            The number of element to include in the network dictionary.

        MCMC_iterations: int
            The number of monte carlo markov chain iterations to run
            for sampling the network during learning.

        sample_size: int
           Number of sample patches that form the minibatch matrix X_t at
           iterations t.

        k: int
            Length of chain motif to use for sampling.

        alpha: int
            By default None. If not none, L1 regularizer for code
            matrix H, which is th solution to the following minimization 
            problem:
                || X - WH||_F^2 + alpha * ||H||_1, 
            where the columns of X contain the sample patches and the columns
            of W form the network dictionary.

        is_glauber_dict: bool
            By default, True. If True, use glauber chain sampling to 
            sample patches during dictionary learning. Otherwise, use 
            pivon chain for sampling.

        is_glauber_recons: bool
            By default, True. If True, use glauber chain sampling to 
            sample patches during network reconstruction. Otherwise, 
            use pivon chain for sampling.

        ONMF_subsample: bool
            By default, True. If True, during the dictionary update step
            from W_{t-1} to W_t, subsample columns of X_t, the sample patches taken
            at iterations t. Else, use the entire matrix X_t.
        
        batch_size: int
             number of patches used for training dictionaries per ONMF iteration.


        omit_folded_edges: bool
            By default, True. If True, ignores edges that are 'folded,' meaning that
            they are already represented within each patch in another entry, caused
            by the MCMC motif folding on itself.

        """
        self.G = G  ### Full netowrk -- could have positive or negagtive edge weights (as a NNetwork or Wtd_NNetwork class)
        if if_tensor_ntwk:
            self.G.set_clrd_edges_signs()
            ### Each edge with weight w is assigned with tensor weight [+(w), -(w)] stored in the field colored_edge_weight

        self.n_components = n_components
        self.MCMC_iterations = MCMC_iterations
        self.sub_iterations = sub_iterations
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.if_tensor_ntwk = if_tensor_ntwk  # if True, input data is a 3d array
        self.omit_folded_edges = omit_folded_edges  # if True, get induced k by k patch without off-chain edges appearing
        ### due to folding of the underlying motif (e.g., completely folded k-chain --> no checkerboard pattern)
        self.W = np.random.rand((k + 1) ** 2, n_components)
        if if_tensor_ntwk:
            self.W = np.random.rand(G.color_dim * (k + 1) ** 2, n_components)

        self.k = k
        self.code = np.zeros(shape=(n_components, sample_size))
        self.code_recons = np.zeros(shape=(n_components, sample_size))
        self.alpha = alpha
        self.is_glauber_dict = is_glauber_dict  ### if false, use pivot chain for dictionary learning
        self.is_glauber_recons = is_glauber_recons  ### if false, use pivot chain for reconstruction
        self.Pivot_exact_MH_rule = Pivot_exact_MH_rule
        self.edges_deleted = []
        self.ONMF_subsample = ONMF_subsample
        self.result_dict = {}
        self.if_wtd_network = if_wtd_network


    def train_dict_RW(self, jump_every=20, verbose=True):
        """
        Performs the Network Dictionary Learning algorithm to train a dictionary
        of latent motifs that aim approximate any given 'patch' of the network.

        Parameters
        ----------
        jump_every: int
            By default, 20. The number of MCMC iterations to perform before
            resampling a patch to encourage coverage of the network.

        verbose: bool
            By default, True. If true, displays a progress bar for training. 

        Returns
        -------
        W: NumPy array, of size k^2 x r.
            The learned dictionary. Each of r columns contains a flattened latent 
            motif of shape k x k. 
        """

        if(verbose):
            print('training dictionaries from patches...')

        G = self.G
        W = self.W
        errors = []
        code = self.code

        if(verbose):
            f = trange
        else:
            f = np.arange

        for t in f(self.MCMC_iterations):
            X = []
            embs = []
            for i in np.arange(self.sample_size):
                S = G.k_node_ind_subgraph(k=self.k)
                while S is None:
                    S = G.k_node_ind_subgraph(k=self.k)
                Adj = S.get_adjacency_matrix()
                X.append(Adj.reshape(1,-1))
                embs.append(list(S.nodes()))
            X = np.asarray(X)
            X = X[:,0,:].T
            if not self.if_tensor_ntwk:
                X = np.expand_dims(X, axis=1)  ### X.shape = (k**2, 1, sample_size)
            if t == 0:
                self.ntf = Online_NMF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      alpha=self.alpha,
                                      mode=2,
                                      learn_joint_dict=True,
                                      subsample=self.ONMF_subsample)  # max number of possible patches
                self.W, self.At, self.Bt, self.Ct, self.H = self.ntf.train_dict()
                self.H = code
            else:
                self.ntf = Online_NMF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      ini_dict=self.W,
                                      ini_A=self.At,
                                      ini_B=self.Bt,
                                      ini_C=self.Ct,
                                      alpha=self.alpha,
                                      history=self.ntf.history,
                                      subsample=self.ONMF_subsample,
                                      mode=2,
                                      learn_joint_dict=True)
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                self.W, self.At, self.Bt, self.Ct, self.H = self.ntf.train_dict()
                code += self.H
                error = np.trace(self.W @ self.At @ self.W.T) - 2 * np.trace(self.W @ self.Bt) + np.trace(self.Ct)
                errors.append(error)
        self.code = code
        self.result_dict.update({'Dictionary learned': self.W})
        self.result_dict.update({'Motif size': self.k})
        self.result_dict.update({'Code learned': self.code})
        self.result_dict.update({'Code COV learned': self.At})
        # print(self.W)
        return self.W

    def display_dict_RW(self,
                     title="Dictionary",
                     path=None,
                     show=True,
                     make_first_atom_2by2=False,
                     show_importance=False):

        """
        Displays the learned dictionary, stored in self.W

        Parameters
        ----------
        title: str
            The title for the plot of the dictionary elements

        path: str
            By defualt, None. If not None, the path in which to 
            save the dictionary plot. 

        show: bool
            By default, True. Whether to show the dictionary plot,
            using plt.show()

        make_first_atom_2by2: bool
            By default, None. If True, increase the size of the top
            atom to emphasize it, as it has the highest 'importance'

        show_importance: bool
            By defualt, False. If True, list the 'importance' of the
            dictionary element under each element, calculated based
            on the code matrix H. 
        """

        W = self.W
        n_components = W.shape[1]
        rows = np.round(np.sqrt(n_components))
        rows = rows.astype(int)
        if rows ** 2 == n_components:
            cols = rows
        else:
            cols = rows + 1


        k = self.k

        ### Use the code covariance matrix At to compute importance
        importance = np.sqrt(self.At.diagonal()) / sum(np.sqrt(self.At.diagonal()))
        # importance = np.sum(self.code, axis=1) / sum(sum(self.code))
        idx = np.argsort(importance)
        idx = np.flip(idx)

        if make_first_atom_2by2:
            ### Make gridspec
            fig = plt.figure(figsize=(3, 6), constrained_layout=False)
            gs1 = fig.add_gridspec(nrows=rows, ncols=cols, wspace=0.1, hspace=0.1)

            for i in range(rows * cols - 3):
                if i == 0:
                    ax = fig.add_subplot(gs1[:2, :2])
                elif i < 2 * cols - 3:  ### first two rows of the dictionary plot
                    if i < cols - 1:
                        ax = fig.add_subplot(gs1[0, i + 1])
                    else:
                        ax = fig.add_subplot(gs1[1, i - (cols - 1) + 2])
                else:
                    i1 = i + 3
                    a = i1 // cols
                    b = i1 % cols
                    ax = fig.add_subplot(gs1[a, b])

                ax.imshow(self.W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                # ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
                # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                ax.set_xticks([])
                ax.set_yticks([])

            plt.suptitle(title)
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
            if type(path) != type(None):
                fig.savefig(path)

            if show:
                plt.show()
 

        else:
            if not self.if_tensor_ntwk:
                figsize = (5, 5)
                if show_importance:
                    figsize = (5, 6)

                fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize,
                                        subplot_kw={'xticks': [], 'yticks': []})

                k = self.k  # number of nodes in the motif F
                for ax, j in zip(axs.flat, range(n_components)):
                    ax.imshow(self.W.T[idx[j]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                    if show_importance:
                        ax.set_xlabel('%1.2f' % importance[idx[j]], fontsize=13)  # get the largest first
                        ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                    # use gray_r to make black = 1 and white = 0

                plt.suptitle(title)
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)

                if type(path) != type(None):
                    fig.savefig(path)
                
                if show:
                    plt.show()

            else:
                W = W.reshape(k ** 2, self.G.color_dim, self.n_components)
                for c in range(self.G.color_dim):
                    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5, 5),
                                            subplot_kw={'xticks': [], 'yticks': []})

                    for ax, j in zip(axs.flat, range(n_components)):
                        ax.imshow(W[:, c, :].T[j].reshape(k, k), cmap="gray_r", interpolation='nearest')
                        # use gray_r to make black = 1 and white = 0
                        if show_importance:
                            ax.set_xlabel('%1.2f' % importance[idx[j]], fontsize=13)  # get the largest first
                            ax.xaxis.set_label_coords(0.5,
                                                      -0.05)  # adjust location of importance appearing beneath patches

                plt.suptitle(title)
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)

                if type(path) != type(None):
                    fig.savefig(path)
                
                if show:
                    plt.show()