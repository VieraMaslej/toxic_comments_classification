import pandas as pd
import numpy as np
import string
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Input, Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, BatchNormalization, SpatialDropout1D, GlobalAveragePooling1D, concatenate, Activation, LSTM, Bidirectional
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD, Adagrad, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from keras.wrappers.scikit_learn import KerasClassifier
import collections

embed_size = 200 
max_features = 20000 
maxlen = 200 

print('Loading data...')
train = pd.read_csv(r"train.csv")
EMBEDDING_FILE = f'glove.twitter.27B.200d.txt' 

classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
yy = train[classes].values
train_size = train["comment_text"].fillna("fillna")

train_sentences, X_test, y, y_test = train_test_split(train_s, yy, train_size=0.1, random_state=42)

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_sentences))
tokenized_train_sentences = tokenizer.texts_to_sequences(train_sentences)
train_padding = pad_sequences(tokenized_train_sentences, maxlen)

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf8'))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


def lstm_cnn_model(dropout_rate=0.1, activation='relu', optimizer='adam'):
    inputs = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inputs)
    x = SpatialDropout1D(dropout_rate)(x)
    x = Bidirectional(LSTM(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
    x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dense(64, activation = activation)(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['acc'])

    return model

model = KerasClassifier(build_fn=lstm_cnn_model, epochs=5) 

import sys
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() 
    def flush(self) :
        for f in self.files:
            f.flush()

# Use scikit-learn to grid search 
activation = ['relu', 'tanh'] 
dropout_rate = [0.1, 0.2]
batch_size = [16, 32, 64]
optimizer = ['SGD', 'RMSprop', 'Adam']
f = open('out.txt', 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)

param_grid = dict(batch_size=batch_size, activation=activation, dropout_rate=dropout_rate, optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_result = grid.fit(train_padding, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means_acc = grid_result.cv_results_['mean_test_score']
stds_acc = grid_result.cv_results_['std_test_score']
params_acc = grid_result.cv_results_['params']

for  means_acc, stds_acc, params_acc in zip(means_acc, stds_acc, params_acc):
    print("%f (%f) with: %r" % (means_acc, stds_acc, params_acc))

