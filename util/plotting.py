import numpy as np
from tqdm import trange

from NNetwork import NNetwork as nn
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn import svm
from sklearn import metrics, model_selection
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


def display_graphs(title,
                   save_path,
                   grid_shape=[2,3],
                   fig_size=[10,10],
                   data = None, # [X, embs]
                   show_importance=False):

        # columns of X = vectorized k x k adjacency matrices
        # corresponding list in embs = sequence of nodes (may overalp)
        X, embs = data
        print('X.shape', X.shape)

        rows = grid_shape[0]
        cols = grid_shape[1]

        fig = plt.figure(figsize=fig_size, constrained_layout=False)
        # make outer gridspec

        idx = np.arange(X.shape[1])
        outer_grid = gridspec.GridSpec(nrows=rows, ncols=cols, wspace=0.02, hspace=0.02)

        # make nested gridspecs
        for i in range(rows * cols):
            a = i // cols
            b = i % rows

            Ndict_wspace = 0.05
            Ndict_hspace = 0.05

            # display graphs
            inner_grid = outer_grid[i].subgridspec(1, 1, wspace=Ndict_wspace, hspace=Ndict_hspace)

            # get rid of duplicate nodes
            A = X[:,idx[i]]
            A = X[:,idx[i]].reshape(int(np.sqrt(X.shape[0])), -1)
            H = nn.NNetwork()
            H.read_adj(A, embs[idx[i]])
            A_sub = H.get_adjacency_matrix()

            # read in as a nx graph for plotting
            G1 = nx.from_numpy_matrix(A_sub)
            ax = fig.add_subplot(inner_grid[0, 0])
            pos = nx.spring_layout(G1)
            edges = G1.edges()
            weights = [1*G1[u][v]['weight'] for u,v in edges]
            nx.draw(G1, with_labels=False, node_size=20, ax=ax, width=weights, label='Graph')

            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(title, fontsize=15)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        fig.savefig(save_path, bbox_inches='tight')
        
        
def display_dictionary(W, dictionary_shape=None ,save_name=None, score=None, grid_shape=None, figsize=[10,10]):
    
    if dictionary_shape is None:
        k = int(np.sqrt(W.shape[0]))
        dict_shape = (k,k)
    else:
        dict_shape = dictionary_shape
        
    rows = int(np.sqrt(W.shape[1]))
    cols = int(np.sqrt(W.shape[1]))
    if grid_shape is not None:
        rows = grid_shape[0]
        cols = grid_shape[1]
    
    figsize0=figsize
    if (score is None) and (grid_shape is not None):
        figsize0=(cols, rows)
    if (score is not None) and (grid_shape is not None):
        figsize0=(cols, rows+0.2)
    
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize0,
                            subplot_kw={'xticks': [], 'yticks': []})
        
        
    for ax, i in zip(axs.flat, range(100)):
        if score is not None:
            idx = np.argsort(score)
            idx = np.flip(idx)    
            
            ax.imshow(W.T[idx[i]].reshape(dict_shape), cmap="viridis", interpolation='nearest')
            ax.set_xlabel('%1.2f' % score[i], fontsize=13)  # get the largest first
            ax.xaxis.set_label_coords(0.5, -0.05)
        else: 
            ax.imshow(W.T[i].reshape(dict_shape), cmap="viridis", interpolation='nearest')
            if score is not None:
                ax.set_xlabel('%1.2f' % score[i], fontsize=13)  # get the largest first
                ax.xaxis.set_label_coords(0.5, -0.05)
       
    plt.tight_layout()
    # plt.suptitle('Dictionary learned from patches of size %d' % k, fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    
    if save_name is not None:
        plt.savefig( save_name, bbox_inches='tight')
    plt.show()
    
    
def display_dict_and_graph(title=None,
                             save_path=None,
                             grid_shape=None,
                             fig_size=[10,10],
                             W = None,
                             regression_coeff = None,
                             At = None,
                             weight_graph = 100,
                             plot_graph_only=False,
                             show_importance=False):
        n_components = W.shape[1]
        k = int(np.sqrt(W.shape[0]))
        rows = np.round(np.sqrt(n_components))
        rows = rows.astype(int)
        if grid_shape is not None:
            rows = grid_shape[0]
            cols = grid_shape[1]
        else:
            if rows ** 2 == n_components:
                cols = rows
            else:
                cols = rows + 1
        if At is None:
            idx = np.arange(W.shape[1])
            # reg_0 = regression_coeff[0, :]
            # idx = np.argsort(reg_0)
            # idx = np.flip(idx)
        else:
            importance = np.sqrt(At.diagonal()) / sum(np.sqrt(At.diagonal()))
            # importance = np.sum(self.code, axis=1) / sum(sum(self.code))
            # importance = regression_coeff
            idx = np.argsort(importance)
            idx = np.flip(idx)
            # reg_0 = regression_coeff[0, :]
            # importance = reg_0
            # idx = np.argsort(importance)
            # idx = np.flip(idx)
        Ndict_wspace = 0.05
        Ndict_hspace = 0.05
        fig = plt.figure(figsize=fig_size, constrained_layout=False)
        ncols = 2
        if plot_graph_only:
            ncols =1
        outer_grid = gridspec.GridSpec(nrows=1, ncols=ncols, wspace=0.02, hspace=0.05)
        for t in np.arange(ncols):
            # make nested gridspecs
            if t == 1:
                ### Make gridspec
                inner_grid = outer_grid[1].subgridspec(rows, cols, wspace=Ndict_wspace, hspace=Ndict_hspace)
                #gs1 = fig.add_gridspec(nrows=rows, ncols=cols, wspace=0.05, hspace=0.05)
                for i in range(rows * cols):
                    a = i // cols
                    b = i % cols
                    ax= fig.add_subplot(inner_grid[a, b])
                    ax.imshow(W.T[idx[i]].reshape(k, k), cmap="gray_r", interpolation='nearest')
                    # ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
                    # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                    ax.set_xticks([])
                    ax.set_yticks([])
            if t == 0:
                inner_grid = outer_grid[0].subgridspec(rows, cols, wspace=Ndict_wspace, hspace=Ndict_hspace)
                #gs1 = fig.add_gridspec(nrows=rows, ncols=cols, wspace=0.05, hspace=0.05)
                for i in range(rows * cols):
                    a = i // cols
                    b = i % cols
                    ax = fig.add_subplot(inner_grid[a, b])
                    if regression_coeff is not None:
                        # ax.set_title(str(np.round(reg_0[i+1], 8)))
                        # ax.set_title(str(np.round(regression_coeff[i+1], 3)))

                        # Get the regression coefficient for this subplot
                        coeff = regression_coeff[i+1]
                        # Base color (blue in different shades)
                        base_color = plt.cm.viridis  # Blues    
                        # Normalize the coefficient for color mapping
                        # Make sure that larger coefficients get a larger value after normalization
                        min_limit = np.percentile(regression_coeff, 10)  # Tighter range for better contrast
                        max_limit = np.percentile(regression_coeff, 90)
                        norm = plt.Normalize((min_limit), np.max(max_limit))
                        # Create a bar in the subplot's title position
                        ax2 = ax.inset_axes([0.1, 1, 1, 0.1])  # Adjust these dimensions as needed
                        # Remove x-axis details from the title bar
                        ax2.axis('off')
                        for j, value in enumerate(zip(coeff)):
                            color_value = norm(value)  # Normalize based on numerical value
                            ax2.bar(j, 1, color=base_color(color_value+0.2), align='center', width=1)
                            ax2.set_xticks([])  # Remove x ticks
                            ax2.set_yticks([])  # Remove y ticks
                            description = str(j+1) + ": " + str(np.round(coeff[j], 2))
                            ax2.text(j, 0.5, description, ha='center', va='center', color='white', fontsize=12)
                            # ax2.spines['top'].set_visible(False)
                            # ax2.spines['right'].set_visible(False)
                            # ax2.spines['bottom'].set_visible(False)
                            # ax2.spines['left'].set_visible(False)
        
                    k = int(np.sqrt(W.shape[0]))
                    A_sub = W[:,idx[i]].reshape(k,k)
                    H = nx.from_numpy_array(A_sub)
                    G1 = nx.Graph()
                    for a in np.arange(k):
                        for b in np.arange(k):
                            u = list(H.nodes())[a]
                            v = list(H.nodes())[b]
                            if H.has_edge(u,v):
                                if np.abs(a-b) == 1:
                                    G1.add_edge(u,v, color='r', weight=A_sub[a,b])
                                else:
                                    G1.add_edge(u,v, color='b', weight=A_sub[a,b])
                    pos = nx.spring_layout(G1)
                    edges = G1.edges()
                    colors = [G1[u][v]['color'] for u,v in edges]
                    weights = [weight_graph*G1[u][v]['weight'] for u,v in edges]
                    nx.draw(G1, with_labels=False, node_size=20, ax=ax, width=weights, edge_color=colors, label='Graph')
                    
                    ## Fix this
                    if show_importance:
                        ax.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
                        ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
#                     ax.set_xticks([])
#                     ax.set_yticks([])

        if title is not None:
            plt.suptitle(title, fontsize=25)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
            