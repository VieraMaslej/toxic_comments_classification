 import pandas as pd
import numpy as np
import string
import os
import re, string
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
from sklearn.feature_extraction.text import TfidfVectorizer

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

# pre-processing
#re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
#def tokenize(s): return re_tok.sub(r' \1 ', s).split()
#tfidf = TfidfVectorizer(max_features=200, tokenizer=tokenize, sublinear_tf=True, min_df=1, norm='l2', encoding='utf-8', lowercase=True , ngram_range=(1, 2), strip_accents='ascii', stop_words='english')

tfidf = TfidfVectorizer(max_features=200, tokenizer=tokenize, sublinear_tf=True, min_df=1, norm='l2', encoding='utf-8', lowercase=False , ngram_range=(1, 2))

X_train = tfidf.fit_transform(train_sentences).toarray()
X_test = tfidf.transform(test_sentences)
        
# cross validation        
kfold = KFold(n_splits=10)
cvscores = []
accscores = []
rocscorec = []

for train, test in kfold.split(X_train, y_train):
    
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

    saved_model = "model.hdf5"
    checkpoint = ModelCheckpoint(saved_model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    print('Training model...')
    history = model.fit(X_train, y_train, batch_size=32, epochs=5, callbacks=[checkpoint], validation_split=0.1)

    scores = model.evaluate(X_train[test], y_train[test])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

    print("Loading model....")
    model = load_model('model.hdf5')
    y_pred = model.predict(X_test)

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
