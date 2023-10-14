# -*- coding: utf-8 -*-

import networkx as nx
import random
import numpy as np
import sys
import csv
from collections import defaultdict
import sqlite3
from itertools import combinations
from tqdm import tqdm

#graph created from DB

ONLY_PL = len(sys.argv) >= 2 and sys.argv[1] == "PL"
GRAPH_PATH = 'data/wordnet_pl.dat' if ONLY_PL else 'data/wordnet_pl_en.dat'


def create_graph():
    conn = sqlite3.connect("data/wordnet.db")
    c = conn.cursor()

    query = """
    select distinct sr.parent_id, 
        sr.child_id,
        sr.rel_id
    from synsetrelation sr 
    """

    if ONLY_PL:
        query = """
        select distinct sr.parent_id, 
            sr.child_id,
            sr.rel_id 
        from synsetrelation sr 
        join synset sp
            on sp.id=sr.parent_id
        join unitandsynset up
            on sp.id=up.syn_id
        join lexicalunit lp 
            on up.lex_id=lp.id
            and lp.pos < 5
        join synset sc
            on sc.id=sr.child_id
        join unitandsynset uc
            on sc.id=uc.syn_id
        join lexicalunit lc
            on uc.lex_id=lc.id
            and lc.pos < 5
        """
    c.execute(query)

    wordnet_graph = nx.MultiDiGraph()
    for (parent_synset_id,
        child_synset_id,
        relation_id
        ) in tqdm(c):
        ps = 's{}'.format(parent_synset_id)
        cs = 's{}'.format(child_synset_id)
        if not ps in wordnet_graph:
            wordnet_graph.add_node(ps)
        if not cs in wordnet_graph:
            wordnet_graph.add_node(cs)
        wordnet_graph.add_edge(ps, cs, relation_id='rSS_{}'.format(relation_id))

    query = """
        select distinct lex_id, syn_id
        from unitandsynset u 
        join lexicalunit l 
            on u.lex_id=l.id
        """

    if ONLY_PL:
        query = """
        select distinct lex_id, syn_id
        from unitandsynset u 
        join lexicalunit l 
            on u.lex_id=l.id
            and l.pos < 5
        """

    c.execute(query)
    synsets = defaultdict(set)
    for (lex_id,
        syn_id
        ) in tqdm(c):
        l = 'l{}'.format(lex_id)
        s = 's{}'.format(syn_id)
        synsets[s].add(l)
        if not l in wordnet_graph:
            wordnet_graph.add_node(l)
        if not s in wordnet_graph:
            wordnet_graph.add_node(s)
        wordnet_graph.add_edge(l, s, relation_id='rLS')
        wordnet_graph.add_edge(s, l, relation_id='rSL')

    for k, v in tqdm(synsets.items()):
        for (l1, l2) in combinations(v, 2):
            wordnet_graph.add_edge(l1, l2, relation_id='rLL')
            wordnet_graph.add_edge(l2, l1, relation_id='rLL')

    query = """
    select distinct parent_id, child_id, rel_id
    from lexicalrelation lr 
    join lexicalunit l1 
        on lr.parent_id=l1.id
    join lexicalunit l2
        on lr.child_id=l2.id
    """

    if ONLY_PL:
        query = """
            select distinct parent_id, child_id, rel_id
            from lexicalrelation lr 
            join lexicalunit l1 
                on lr.parent_id=l1.id
            join lexicalunit l2
                on lr.child_id=l2.id
            and l1.pos < 5
            and l2.pos < 5  
            """

    c.execute(query)
    for (parent_id,
        child_id,
        rel_id
        ) in tqdm(c):
        p = 'l{}'.format(parent_id)
        c = 'l{}'.format(child_id)
        if not p in wordnet_graph:
            wordnet_graph.add_node(p)
        if not c in wordnet_graph:
            wordnet_graph.add_node(c)
        wordnet_graph.add_edge(p, c, relation_id='rLL_{}'.format(rel_id))
    nx.write_gpickle(wordnet_graph, GRAPH_PATH, 4)

create_graph()
