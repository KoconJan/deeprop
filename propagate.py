import pickle
from collections import defaultdict
import operator
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, classification_report
import numpy as np
import copy
import pandas as pd
import tensorflow as tf
#from keras.layers.advanced_activations import PReLU
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.nn import sigmoid_cross_entropy_with_logits
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, ActivityRegularization, Conv1D, Reshape, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop, Adam
import sys
from tensorflow.keras.utils import Sequence

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


INPUT_DIM=300
OUTPUT_DIM=26
FOLD_ID = 1
if len(sys.argv) >= 2:
    FOLD_ID = int(sys.argv[1])


def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """

    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost)  # average on all labels

    return macro_cost


def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost

def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

def loss_coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return SS_res / (SS_tot)

def loss_fn(y_true, y_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

def abs_KL_div(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), None)
    y_pred = K.clip(y_pred, K.epsilon(), None)
    return K.sum( K.abs( (y_true- y_pred) * (K.log(y_true / y_pred))), axis=-1)

def baseline_model():
    model = Sequential()
    model.add(Dense(INPUT_DIM, input_dim=INPUT_DIM, kernel_initializer='normal', activation='relu'))
    model.add(Dense(OUTPUT_DIM, kernel_initializer='normal'))
    model.compile(loss=BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    return model

def baseline_model_class():
    model = Sequential()
    model.add(Dense(INPUT_DIM, input_dim=INPUT_DIM, kernel_initializer='normal', activation='relu'))
    model.add(Dense(OUTPUT_DIM, activation='sigmoid'))
    model.compile(loss=macro_double_soft_f1, optimizer='adam', metrics=[macro_f1])
    return model

def deep_model_class():
    model = Sequential()
    model.add(Dense(INPUT_DIM, input_dim=INPUT_DIM, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(INPUT_DIM, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(INPUT_DIM,  activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(INPUT_DIM,  activation='relu'))
    model.add(Dense(OUTPUT_DIM, activation='sigmoid'))
    model.compile(loss=macro_double_soft_f1, optimizer='adam', metrics=[macro_f1])
    return model

def deep_model():
    model = Sequential()
    model.add(Dense(INPUT_DIM, input_dim=INPUT_DIM, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(OUTPUT_DIM, kernel_initializer='normal', activation='linear'))
    model.compile(loss=loss_coeff_determination, optimizer='adam')
    return model

def deep_model2():
    model_m = Sequential()
    model_m.add(Reshape((20, 15), input_shape=(INPUT_DIM,)))
    model_m.add(Conv1D(300, 5, activation='relu', input_shape=(20, 15)))
    model_m.add(Conv1D(300, 5, activation='relu'))
    model_m.add(MaxPooling1D(3))
    #model_m.add(Conv1D(500, 4, activation='relu'))
    model_m.add(GlobalAveragePooling1D())
    model_m.add(Dropout(0.5))
    model_m.add(Dense(OUTPUT_DIM, activation='linear'))
    model_m.compile(loss=loss_coeff_determination, optimizer='adam')
    return model_m


class SequenceExample(Sequence):

    def __init__(self, x_in, y_in, batch_size, shuffle=True):
        # Initialization
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = x_in
        self.y = y_in
        self.datalen = len(y_in)
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x_batch = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        return x_batch, y_batch

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)

# WORDNET_PICKLE_PATH = 'wordnet_pleng_neighbours.dat'
WORDNET_PICKLE_PATH = 'neighbours/wordnet_pl_neighbours.dat'
# WORDNET_PICKLE_PATH = 'neighbours/wordnet_pl_en_neighbours.dat'

# WORDNET_DATA_FOLDS_PATH = 'wordnet_emo_data_folds.dat'
WORDNET_DATA_FOLDS_PATH = 'data_folds/wordnet_pl_emo_data_folds.dat'
# WORDNET_DATA_FOLDS_PATH = 'data_folds/wordnet_pl_en_emo_data_folds.dat'


def get_n_nns(emo_ids, wordnet_dict):
    s = set(emo_ids)
    nns_set = set()
    for emo_id in emo_ids:
        for item_id in wordnet_dict[emo_id]:
            if not item_id in s:
                nns_set.add(item_id)
    return nns_set


with open(WORDNET_PICKLE_PATH, 'rb') as f:
    wordnet_dict = pickle.load(f)
with open(WORDNET_DATA_FOLDS_PATH, 'rb') as f:
    emo_dict = pickle.load(f)
    folds_list = pickle.load(f)
    emo_embeddings = pickle.load(f)

print(emo_dict['##headers'])
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
i = 0


for train_ids, test_ids in folds_list:
    i += 1
    print('Fold: ', i)
    if i != FOLD_ID:
        continue
    X_train, y_train, X_test, y_test, new_train_ids, new_test_ids = [], [], [], [], [], []
    for train_id in train_ids:
        #print(train_id)
        #print(list(emo_embeddings.keys())[:10])
        #print(list(emo_dict.keys())[:10])

        if train_id in emo_embeddings and train_id in emo_dict:
            #print('here')
            X_train.append(emo_embeddings[train_id])
            y_train.append(emo_dict[train_id])
            new_train_ids.append(train_id)
    for test_id in test_ids:
        if test_id in emo_embeddings and test_id in emo_dict:
            X_test.append(emo_embeddings[test_id])
            y_test.append(emo_dict[test_id])
            new_test_ids.append(test_id)
    new_ids = get_n_nns(new_train_ids, wordnet_dict)
    X_train, y_train, X_test, y_test = pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test), pd.DataFrame(
        y_test)
    #s = SequenceExample(X_train.to_numpy(), y_train.to_numpy(), batch_size=1024)
    initial_train_size = len(X_train)
    ii = 0
    while len(new_ids) > 0 and ii < 5:
        ii += 1
        print('Iteration: ', ii)
        print('Batch size: ', int(1024*(len(X_train)/initial_train_size)))
        #model = deep_model()
        model = deep_model_class()
        
        #model = baseline_model_class()
        es = EarlyStopping(monitor='val_macro_f1', mode='max', restore_best_weights=True, patience=30)
        model.fit(X_train, y_train, epochs=300, batch_size=int(1024*(len(X_train)/initial_train_size)), verbose=2, validation_data=(X_test, y_test),
                   callbacks=[es], workers=4, use_multiprocessing=True)
        #model.fit(s, epochs=300, batch_size=1024, verbose=2, validation_data=(X_test, y_test),
        #          callbacks=[es], workers=6)
        y_pred = model.predict(X_train)
        #print('X_train: ', r2_score(y_train, y_pred), mean_squared_error(y_train, y_pred), f1_score(y_train, y_pred.round(), average='macro'))
        print('X_train: ', f1_score(y_train, y_pred.round(), average='macro'))
        y_pred = model.predict(X_test)
        #print('X_test', r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred), f1_score(y_test, y_pred.round(), average='macro'))
        print('X_test', f1_score(y_test, y_pred.round(), average='macro'), f1_score(y_test, y_pred.round(), average='micro'))
        print(classification_report(y_test, y_pred.round()))
        X_new = []
        for new_id in new_ids:
            X_new.append(emo_embeddings[new_id])
            new_train_ids.append(new_id)
        X_new = pd.DataFrame(X_new)
        y_new = pd.DataFrame(model.predict(X_new))
        X_train = X_train.append(X_new)
        y_train = y_train.append(y_new.round())
        pd.DataFrame(new_train_ids).to_hdf('train_pl_out/fold_{}_checkpoint_{}.hdf5'.format(i, ii), 'train_ids', mode='w')
        y_train.to_hdf('train_pl_out/fold_{}_checkpoint_{}.hdf5'.format(i, ii), 'y_train')
        new_ids = get_n_nns(new_train_ids, wordnet_dict)
