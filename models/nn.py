import pandas as pd
import numpy as np
import string

from keras.layers import Embedding
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence importos pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout, Conv1D, GlobalMaxPooling1D, MaxPooling1D, SpatialDropout1D, GlobalAveragePooling1D, Input, concatenate, LSTM, Bidirectional
from keraslayers import 
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import model_from_json

embed_size = 300
max_features = 20000 
maxlen = 200 

print('Loading data...')
train = pd.read_csv(r"train.csv")
test = pd.read_csv(r"test.csv")
test_labels = pd.read_csv(r"test_labels.csv")

EMBEDDING_FILE = f'glove.840B.300d.txt' 

test = pd.concat([test, test_labels], axis=1)
test = test[test['toxic']!=-1]

classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[classes].values
y_test = test[classes].values

train_sentences = train["comment_text"].fillna("fillna")
test_sentences = test["comment_text"].fillna("fillna")

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_sentences))

tokenized_train_sentences = tokenizer.texts_to_sequences(train_sentences)
tokenized_test_sentences = tokenizer.texts_to_sequences(test_sentences)

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
    
#####################################################################################
#                                 FFNN                                              #
#####################################################################################

    inputs = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inputs)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)

################################################################################

    saved_model = "model.hdf5"
    checkpoint = ModelCheckpoint(saved_model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    print('Training model...')
    history = model.fit(train_padding, y, batch_size=32, epochs=5, callbacks=[checkpoint], validation_split=0.1)

    scores = model.evaluate(train_padding[test], y[test])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

    print("Loading model....")
    model = load_model('model.hdf5')
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
print("Test accuracy is: {} %.2f (+/- %.2f)" %  (np.mean(accscores), np.std(accscores)))
print("Test roc-auc is: {} %.2f (+/- %.2f)" % (np.mean(rocscores), np.std(rocscores)))


#####################################################################################
#                                 MODELS                                            #
#####################################################################################
#                              1. CNN                                               #
#####################################################################################

#   inputs = Input(shape=(maxlen,))
#   x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inputs)
#   x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
#   x = MaxPooling1D(pool_size=2)(x)
#   x = Flatten()(x)
#   x = Dropout(0.2)(x)
#   x = Dense(128, activation='relu')(x)
#   output = Dense(6, activation='sigmoid')(x)
#   model = Model(inputs=inputs, outputs=output)

#####################################################################################
#                              2. GRU                                               #
#####################################################################################

#   inputs = Input(shape=(maxlen,))
#   x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inputs)
#   x = GRU(128)(x)
#   x = Flatten()(x)
#   x = Dropout(0.2)(x)
#   x = Dense(128, activation='relu')(x)
#   output = Dense(6, activation='sigmoid')(x)
#   model = Model(inputs=inputs, outputs=output)

#####################################################################################
#                              3. LSTM                                              #
#####################################################################################

#   inputs = Input(shape=(maxlen,))
#   x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inputs)
#   x = LSTM(128)(x)
#   x = Flatten()(x)
#   x = Dropout(0.2)(x)
#   x = Dense(128, activation='relu')(x)
#   output = Dense(6, activation='sigmoid')(x)
#   model = Model(inputs=inputs, outputs=output)

#####################################################################################
#                              4. biGRU+CNN                                         #
#####################################################################################

#   inputs = Input(shape=(maxlen,))
#   x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inputs)
#   x = SpatialDropout1D(0.2)(x)
#   x = Bidirectional(GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
#   x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
#   avg_pool = GlobalAveragePooling1D()(x)
#   max_pool = GlobalMaxPooling1D()(x)
#   x = concatenate([avg_pool, max_pool])
#   x = Dense(64, activation='relu')(x)
#   x = Dropout(0.2)(x)
#   output = Dense(6, activation='sigmoid')(x)
#   model = Model(inputs=inputs, outputs=output)


#####################################################################################
#                              5. biLSTM+CNN                                        #
#####################################################################################

#    inputs = Input(shape=(maxlen,))
#    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inputs)
#    x = SpatialDropout1D(0.2)(x)
#    x = Bidirectional(LSTM(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
#    x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
#    avg_pool = GlobalAveragePooling1D()(x)
#    max_pool = GlobalMaxPooling1D()(x)
#    x = concatenate([avg_pool, max_pool])
#    x = Dense(64, activation='relu')(x)
#    x = Dropout(0.2)(x)
#    output = Dense(6, activation='sigmoid')(x)
#    model = Model(inputs=inputs, outputs=output)




