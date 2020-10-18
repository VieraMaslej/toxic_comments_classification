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
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold


embed_size = 300
max_features = 20000 
maxlen = 200 

print('Loading data...')
train = pd.read_csv(r"train.csv")
test = pd.read_csv(r"test.csv")
test_labels = pd.read_csv(r"test_labels.csv")

test = pd.concat([test, test_labels], axis=1)
test = test[test['toxic']!=-1]

classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = train[classes].values
y_test = test[classes].values

train_sentences = train["comment_text"].fillna("fillna")
test_sentences = test["comment_text"].fillna("fillna")

print('Preprocessing train') 
train = list()
for i in train_sentences:
    tokens = word_tokenize(i)
    tokens = [w.lower() for w in tokens] 
    table= str.maketrans('','', string.punctuation) 
    stripped=[w.translate(table) for w in tokens]
    word = [w for w in stripped if w.isalpha()] 
    stop_words=set(stopwords.words('english'))
    word = [w for w in word if not w in stop_words]
    train.append(word)

print('Preprocessing test')
test = list()
for i in test_sentences:
    tokens = word_tokenize(i)
    tokens = [w.lower() for w in tokens]
    table= str.maketrans('','', string.punctuation)
    stripped=[w.translate(table) for w in tokens] 
    word = [w for w in stripped if w.isalpha()]
    stop_words=set(stopwords.words('english'))
    word = [w for w in word if not w in stop_words] 
    test.append(word) 

# 1. create custom embeddings 
words = list()
for i in train_sentences:
    tokens = word_tokenize(i) # word tokenizacia
    words.append(tokens) # pridá výsledné slová do prázdneho listu, ktorý sme na začiatku vytvorili
print('Word2Vec...') 

from gensim.models import Word2Vec
from keras.layers import Embedding

model = Word2Vec(words, min_count = 1) #word to vector, implementacia z kniznice gensim
vocabulary = model.wv.vocab #slovnik
name = 'w2vPP.txt'
model.wv.save_word2vec_format(name, binary = False)

EMBEDDING_FILE = 'w2vPP.txt' # load embeddings


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_sentences))

tokenized_train_sentences = tokenizer.texts_to_sequences(train)
tokenized_test_sentences = tokenizer.texts_to_sequences(test)

train_padding = pad_sequences(tokenized_train_sentences, maxlen)
test_padding = pad_sequences(tokenized_test_sentences, maxlen)

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

# cross validation        
kfold = KFold(n_splits=10)
cvscores = []
accscores = []
rocscorec = []

for train, test in kfold.split(train_padding, y):
    
    inputs = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inputs)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
    x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())

    saved_model = "w2vPP.hdf5"
    checkpoint = ModelCheckpoint(saved_model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    print('Training model...')
    history = model.fit(train_padding, y_train, batch_size=32, epochs=5, callbacks=[checkpoint], validation_split=0.1)

    scores = model.evaluate(train_padding[test], y[test])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

    print("Loading model....")
    model = load_model('w2vPP.hdf5')
    y_pred = model.predict(test_padding)

    y_int = np.zeros_like(y_pred)
    y_int[y_pred > 0.5] = 1

    accuracy = accuracy_score(y_test,y_int)
    print('Accuracy is {}'.format(accuracy))
    accscores.append(accuracy)
    
    rocauc = roc_auc_score(y_test, y_pred)
    print('Roc-auc score is {}'.format(rocauc))
    rocscore.append(rocauc)
    
    print('Classification report {}'.format(classification_report(y_test, y_int, zero_division=0)))
    print('Confusion matrix {}'.format(multilabel_confusion_matrix(y_test, y_int)))
        
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
print("Test accuracy is: {} %.2f%% (+/- %.2f%%)" (np.mean(accscores), np.std(accscores)))
print("Test roc-auc is: {} %.2f%% (+/- %.2f%%)" (np.mean(rocscores), np.std(rocscores)))
