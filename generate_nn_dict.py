import networkx as nx
from collections import defaultdict
import pickle
from tqdm import tqdm

WORDNET_NX_GRAPH_PATH = 'data/wordnet_{}.dat'
WORDNET_PICKLE_PATH = 'neighbours/wordnet_{}_neighbours.dat'



for lang in ['pl', 'pl_en']:
    g = nx.read_gpickle(WORDNET_NX_GRAPH_PATH.format(lang))  # type: nx.MultiDiGraph
    g2 = nx.Graph(g)

    d = defaultdict(set)
    for node in tqdm(g2):
        for neighbour_node in g2[node]:
            d[node].add(neighbour_node)
            d[neighbour_node].add(node)

    with open(WORDNET_PICKLE_PATH.format(lang), 'wb') as f:
        pickle.dump(d, f, -1)
