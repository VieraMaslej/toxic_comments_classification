import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score
from keras.models import load_model

max_features = 20000 
maxlen = 200

# load data
test = pd.read_csv(r"test.csv")
test_labels = pd.read_csv(r"test_labels.csv")
test = pd.concat([test, test_labels], axis=1)
test = test[test['toxic']!=-1]

classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = test[classes].values
test_sentences = test["comment_text"].fillna("fillna")

tokenizer = Tokenizer(num_words=max_features))
tokenized_test_sentences = tokenizer.texts_to_sequences(test_sentences)
test_padding = pad_sequences(tokenized_test_sentences, maxlen)

############# load model #############################################
model = load_model('model_name.hdf5')
######################################################################

# predict
y_pred = model.predict(test_padding)

# evalution
print('Roc-auc score is {}'.format(roc_auc_score(y, y_pred)))

# probabilities to integer
y_int = np.zeros_like(y_pred)
y_int[y_pred > 0.5] = 1

print('Accuracy is {}'.format(accuracy_score(y,y_int)))
print('Classification report {}'.format(classification_report(y, y_int, zero_division=0)))
print('Confusion matrix {}'.format(multilabel_confusion_matrix(y, y_int)))

