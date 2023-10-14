# -*- coding: utf-8 -*-

import networkx as nx
import random
import numpy as np
import sys
import csv
from collections import defaultdict
#import mysql.connector
from itertools import combinations
from tqdm import tqdm

#graph created from DB

ONLY_PL = len(sys.argv) >= 2 and sys.argv[1] == "PL"
GRAPH_PATH = 'data/wordnet_pl.dat' if ONLY_PL else 'data/wordnet_pl_en.dat'
OUTPUT_DIR = 'random_walks_pl' if ONLY_PL else 'random_walks_all'


def generate_random_walk(output_suffix):
    g = nx.read_gpickle(GRAPH_PATH)  # type: nx.MultiDiGraph
    N_WALKS = 1000000
    i = 0
    f = open('{}/random_walks_{}.txt'.format(OUTPUT_DIR, output_suffix), 'w')
    while i < N_WALKS:
        for n in random.sample(g.nodes, 1000):
            stay = True
            node = n
            random_walk = [n]
            random_walk_set = set()
            out_edges = g.out_edges(node, data='relation_id')
            while out_edges and len(random_walk) < 40:
                (node_from, node_to, relation) = random.sample(list(out_edges), 1)[0]
                random_walk.append(relation)
                random_walk.append(node_to)
                random_walk_set.add((node_from, node_to, relation))
                node = node_to
                out_edges = g.out_edges(node, data='relation_id')
            if len(random_walk) > 9 and i < N_WALKS:
                f.write(' '.join(random_walk))
                f.write('\n')
                i += 1
            else:
                print('short')
            if i % 10000 == 0:
                print(i)
    f.close()

def generate_random_walks():
    if len(sys.argv) < 2:
        generate_random_walk(0)
    else:
        generate_random_walk(sys.argv[2])


generate_random_walks()

