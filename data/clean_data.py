import pandas as pd
import networkx as nx

def load_tsv_and_create_graph(tsv_file):
    df = pd.read_csv(tsv_file, sep=' /t', header=None, names=['source', 'target'])
    unique_strings = pd.concat([df['source'], df['target']]).unique()
    string_to_int = {string: idx for idx, string in enumerate(unique_strings)}
    df['source'] = df['source'].map(string_to_int)
    df['target'] = df['target'].map(string_to_int)
    G = nx.from_pandas_edgelist(df, 'source', 'target')
    
    return G, string_to_int

def save_graph_to_txt(G, output_file):
    with open(output_file, 'w') as f:
        for edge in G.edges():
            f.write(f"{edge[0]},{edge[1]}\n")

tsv_file = 'data/miner.tsv'
output_file = 'data/miner.txt'

G, string_to_int_map = load_tsv_and_create_graph(tsv_file)

save_graph_to_txt(G, output_file)

print(string_to_int_map)
