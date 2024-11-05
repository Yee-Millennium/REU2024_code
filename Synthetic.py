import networkx as nx
import numpy as np

# Erdős-Rényi Network (ER)
def ER_synthetic(n, p, save_path='data/ER_n{n}_p{p:.2f}.txt'):
    """
    Generates an Erdős-Rényi (ER) random graph with `n` nodes and connection probability `p`.
    
    Args:
        n (int): Number of nodes.
        p (float): Probability for edge creation.
        save_path (str): File path to save the generated network as an edge list.
    """
    save_path = save_path.format(n=n, p=p)
    G_er = nx.erdos_renyi_graph(n=n, p=p)
    nx.write_edgelist(G_er, save_path, data=False)


# Watts-Strogatz Network (WS)
def WS_synthetic(n, k, p, save_path='data/WS_n{n}_k{k}_p{p:.2f}.txt'):
    """
    Generates a Watts-Strogatz (WS) small-world graph with `n` nodes, each connected to `k` neighbors,
    and rewired with probability `p`.
    
    Args:
        n (int): Number of nodes.
        k (int): Each node is connected to `k` nearest neighbors in ring topology.
        p (float): Probability of rewiring each edge.
        save_path (str): File path to save the generated network as an edge list.
    """
    save_path = save_path.format(n=n, k=k, p=p)
    G_ws = nx.watts_strogatz_graph(n=n, k=k, p=p)
    nx.write_edgelist(G_ws, save_path, data=False)


# Barabási-Albert Network (BA)
def BA_synthetic(n, m, save_path='data/BA_n{n}_m{m}.txt'):
    """
    Generates a Barabási-Albert (BA) scale-free graph with `n` nodes and each new node connected to `m` existing nodes.
    
    Args:
        n (int): Number of nodes.
        m (int): Number of edges to attach from a new node to existing nodes.
        save_path (str): File path to save the generated network as an edge list.
    """
    save_path = save_path.format(n=n, m=m)
    G_ba = nx.barabasi_albert_graph(n=n, m=m)
    nx.write_edgelist(G_ba, save_path, data=False)


# Configuration Model Network (CM) with Two Options for Degree Sequence
import networkx as nx
import numpy as np

def CM_synthetic(n=1000, method="BA", m=5, exponent=2.5, save_path='data/CM_n{n}_method{method}_m{m}{exp}.txt'):
    """
    Generates a Configuration Model (CM) random graph using a degree sequence derived from either:
    - A Barabási-Albert (BA) network
    - A power-law distribution

    Args:
        n (int): Number of nodes.
        method (str): Method for degree sequence, "BA" for Barabási-Albert or "powerlaw" for power-law distribution.
        m (int): Number of edges to attach from a new node in BA (used if method="BA").
        exponent (float): Exponent of the power-law distribution (used if method="powerlaw").
        save_path (str): File path to save the generated network as an edge list.
    """
    if method == "BA":
        # Option 1: Degree sequence from a BA network
        G_ba = nx.barabasi_albert_graph(n=n, m=m)
        degree_sequence = [d for _, d in G_ba.degree()]
        save_path = save_path.format(n=n, method="BA", m=m, exp="")

    elif method == "powerlaw":
        # Option 2: Degree sequence sampled from a power-law distribution
        degree_sequence = np.random.zipf(a=exponent, size=n)
        if sum(degree_sequence) % 2 != 0:
            degree_sequence[-1] += 1
        save_path = save_path.format(n=n, method="powerlaw", m="", exp=f"_exp{exponent}")
    
    else:
        raise ValueError("Invalid method. Choose 'BA' or 'powerlaw'.")

    # Generate the CM graph
    G_cm = nx.configuration_model(degree_sequence)
    G_cm = nx.Graph(G_cm)  # Remove parallel edges by creating a simple graph
    G_cm.remove_edges_from(nx.selfloop_edges(G_cm))  # Remove self-loops
    nx.write_edgelist(G_cm, save_path, data=False)
