import pickle
from sklearn.model_selection import KFold
import numpy as np
import fasttext
from tqdm import tqdm

WORDNET_PICKLE_PATH = 'data/wordnet_{}.dat'
WORDNET_LU_EMOTIONS_PATH = 'data/original_plwordnet_emo_meta.csv'
WORDNET_DATA_FOLDS_PATH = 'data_folds/wordnet_{}_emo_data_folds.dat'
WORDNET_EMBEDDINGS_PATH = 'wordnet_embeddings/random_walks_{}.bin'

for (lang, emb) in [('pl', 'pl'), ('pl_en', 'all')]:
    d = dict()
    keys = []
    with open(WORDNET_LU_EMOTIONS_PATH) as f:
        d['##headers'] = f.readline().strip().split()[1:]
        for line in f:
            items = line.strip().split(',')
            key = 'l{}'.format(items[0])
            d[key] = list(map(float, items[1:]))
            keys.append(key)
    keys = np.array(keys)
    kf = KFold(n_splits=10, shuffle=True, random_state=2)
    folds = []
    for train_index, test_index in kf.split(keys):
        X, Y = keys[train_index], keys[test_index]
        folds.append([list(X), list(Y)])

    with open(WORDNET_PICKLE_PATH.format(lang), 'rb') as f:
        wordnet_dict = pickle.load(f)
    emo_embeddings = dict()

    wordnet_embeddings_model = fasttext.load_model(WORDNET_EMBEDDINGS_PATH.format(emb))
    for item_id in tqdm(wordnet_dict):
        emo_embeddings[item_id] = np.hstack(
            (wordnet_embeddings_model.get_word_vector(item_id)))

    with open(WORDNET_DATA_FOLDS_PATH.format(lang), 'wb') as f:
        pickle.dump(d, f, -1)
        pickle.dump(folds, f, -1)
        pickle.dump(emo_embeddings, f, -1)




