# -*- coding: utf-8 -*-
# Copyright ©2022 Jan Kocoń, Wrocław University of Technology
# jan.kocon@pwr.edu.pl

import random
import numpy as np
import sys
import csv
from collections import defaultdict
import sqlite3
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict as dd
import re
import pandas as pd

affective_map = {
    'bezużyteczność': 'val_nonusefulness',
    'błąd': 'val_mistake',
    'bład': 'val_mistake',
    'brzydota': 'val_ugliness',
    'brzytoda': 'val_ugliness',
    'brzygota': 'val_ugliness',
    'cieszenie się na coś oczekiwanego': 'emo_anticipation',
    'cieszenie się na': 'emo_anticipation',
    'cieszenie sie': 'emo_anticipation',
    'cieszenie sie na': 'emo_anticipation',
    'cieszenie się': 'emo_anticipation',
    'ciesznie się na': 'emo_anticipation',
    'oczekiwanie na': 'emo_anticipation',
    'dobro': 'val_goodness',
    'dobro drugiego': 'val_goodness',
    'dobro drugiego człowieka': 'val_goodness',
    'krzywda': 'val_harm',
    'krzwda': 'val_harm',
    'krzyada': 'val_harm',
    'krzywda błąd': 'val_harm',
    'niewiedza': 'val_ignorance',
    'nieiwedza': 'val_ignorance',
    'nieszczęście': 'val_unhappiness',
    'niedzczęście': 'val_unhappiness',
    'nieszczeście': 'val_unhappiness',
    'nieszczęscie': 'val_unhappiness',
    'nieszczęśćie': 'val_unhappiness',
    'nieużyteczność': 'val_nonusefulness',
    'nieurzyteczność': 'val_nonusefulness',
    'nieuzyteczność': 'val_nonusefulness',
    'nieużytecznosć': 'val_nonusefulness',
    'nieużytecznośc': 'val_nonusefulness',
    'nieużyteczność krzywda': 'val_nonusefulness',
    'nieużyteczość': 'val_nonusefulness',
    'nieżuyteczność': 'val_nonusefulness',
    'nieżyteczność': 'val_nonusefulness',
    'niużyteczność': 'val_nonusefulness',   
    'nie': 'val_nonusefulness', 
    'piękno': 'val_beauty',
    'piekno': 'val_beauty',
    'prawda': 'val_truth',
    'radość': 'emo_joy',
    'radosć': 'emo_joy',
    'radoć': 'emo_joy',
    'radośc': 'emo_joy',    
    'smutek': 'emo_sadness',
    's mutek': 'emo_sadness',
    'smitek': 'emo_sadness',
    'smute': 'emo_sadness',
    'szczęście': 'val_happiness',
    'szczeście': 'val_happiness',
    'sz częście': 'val_happiness',
    'szczęscie': 'val_happiness',
    'szczęści': 'val_happiness',
    'szczęście użyteczność': 'val_happiness',
    'szczęśćie': 'val_happiness',    
    'strach': 'emo_fear',
    'strach wstręt': 'emo_fear',
    'użyteczność': 'val_usefulness',
    'uzyteczność': 'val_usefulness',
    'użytecznośc': 'val_usefulness',
    'użuyteczność': 'val_usefulness',
    'użytecznosć': 'val_usefulness',
    'użyteczność dobro': 'val_usefulness',
    'użyteczność szczęście': 'val_usefulness',
    'użyteczność wiedza': 'val_usefulness',    
    'wiedza': 'val_knowledge',
    'wstręt': 'emo_disgust',
    'wstret': 'emo_disgust',
    'wstrę': 'emo_disgust',
    'wstęt': 'emo_disgust',    
    'zaufanie': 'emo_trust',
    'zaufanie złość': 'emo_trust',
    'zufanie': 'emo_trust',
    'zaskoczenie czymś nieprzewidywanym': 'emo_surprise',
    'zaskoczenie': 'emo_surprise',
    'złość': 'emo_anger',
    'zlość': 'emo_anger',
    'złosć': 'emo_anger',
    'złośc': 'emo_anger',
    'złość wstręt': 'emo_anger',
    'złóść': 'emo_anger',
    'gniew': 'emo_anger',
    '+ m': 'pol_strong_positive',
    '+ s': 'pol_weak_positive',
    '- m': 'pol_strong_negative',
    '- s': 'pol_weak_negative',
    '-': 'pol_weak_negative',
    'amb': 'pol_ambivalent',
    '': 'pol_neutral',
}
vals = ['val_nonusefulness',
    'val_mistake',
    'val_ugliness',
    'emo_anticipation',
    'val_goodness',
    'val_harm',
    'val_ignorance',
    'val_unhappiness',
    'val_beauty',
    'val_truth',
    'emo_joy',
    'emo_sadness',
    'val_happiness',
    'emo_fear',
    'val_usefulness',
    'val_knowledge',
    'emo_disgust',
    'emo_trust',
    'emo_surprise',
    'emo_anger',
    'pol_strong_positive',
    'pol_weak_positive',
    'pol_strong_negative',
    'pol_weak_negative',
    'pol_ambivalent',
    'pol_neutral']


def get_emotions():
    conn = sqlite3.connect("data/wordnet.db")
    c = conn.cursor()

    query = """
    select distinct lexicalunit_id, GROUP_CONCAT(emotions, ';') as emotions, GROUP_CONCAT(valuations, ';') as valuations, GROUP_CONCAT(markedness, ';') as markedness
    from emotion
	group by lexicalunit_id
    """
    
    c.execute(query)
    i = 0
    d = dd(lambda: dd(int))

    l = list()
    l.append(['lexicalunit_id'] + vals)
    for (lexicalunit_id,
        emotions,
        valuations,
        markedness
        ) in tqdm(c):

        for val in vals:
            d[lexicalunit_id][val] = 0
        for values in [emotions, valuations, markedness]:
            if not values:
                d[lexicalunit_id][affective_map['']] = 1
            else:
                isneutral = True
                for x in filter(None, re.split(';|:|\.', values)):
                    try:
                        v = affective_map[x.strip()]
                    except:
                        print("!!!! ", values)
                    if v != 'pol_neutral':
                        isneutral = False
                        d[lexicalunit_id][v] = 1
                if isneutral:
                    d[lexicalunit_id]['pol_neutral'] = 1
        y = []
        for aff in vals:
            val = d[lexicalunit_id][aff] 
            y.append(val)
        if y[-1] == 1 and sum(y[:-1]) > 0:
            y[-1] = 0
            i += 1
        l.append([lexicalunit_id] + y)

    df2 = pd.DataFrame(l)
    new_header = df2.iloc[0]
    df2 = df2[1:]
    df2.columns = new_header
    df2.to_csv('data/original_plwordnet_emo_meta.csv', index=False)


get_emotions()
