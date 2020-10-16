import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import *
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Accuracy
from sklearn.metrics import classification_report, multilabel_confusion_matrix, roc_auc_score, average_precision_score, recall_score, precision_score, matthews_corrcoef, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, accuracy_score

# loading data
data = pd.read_csv(r"train.csv")
test_data = pd.read_csv(r"test.csv")
test_labels = pd.read_csv(r"test_labels.csv")

test_data = test_data.merge(test_labels, how="right")
test_data_new = test_data[test_data['toxic']!=-1].reset_index(drop=True)

classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = test_data_new.drop(["id","comment_text"], axis=1)
y_train = data.drop(["id","comment_text"], axis=1)

train_sentences = data["comment_text"].values
test_sentences = test_data_new["comment_text"].values

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

padded_ids_train = []
mask_ids_train = []

for i in tqdm(range(len(train_sentences))):
    encoding = tokenizer.encode_plus(train_sentences[i], max_length=128, pad_to_max_length=True, do_lower_case= False)
    input_ids, attention_id = encoding["input_ids"], encoding["attention_mask"]
    padded_ids_train.append(input_ids)
    mask_ids_train.append(attention_id)

padded_ids_test = []
mask_ids_test = []

for i in tqdm(range(len(test_sentences))):
    encoding=tokenizer.encode_plus(test_sentences[i], max_length=128, pad_to_max_length=True, do_lower_case= False)
    input_ids, attention_id = encoding["input_ids"], encoding["attention_mask"]
    padded_ids_test.append(input_ids)
    mask_ids_test.append(attention_id)

train_id = np.array(padded_ids_train)
train_mask = np.array(mask_ids_train)
test_id = np.array(padded_ids_test)
test_mask = np.array(mask_ids_test)

input_1 = tf.keras.Input(shape = (128) , dtype=np.int32)
input_2 = tf.keras.Input(shape = (128) , dtype=np.int32)
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')
output  = model([input_1 , input_2] , training = False )
x = tf.keras.layers.Dense(128 , activation = tf.nn.relu )(output[0])  
x = tf.keras.layers.Dropout(0.15)(x)                             
x = tf.keras.layers.Dense(6 , activation = tf.nn.sigmoid )(x)
model = tf.keras.Model(inputs = [input_1, input_2 ] , outputs = [x])
model.summary()

path= "model_bert1.h5"
checkpoint = ModelCheckpoint(filepath=path, monitor='val_precision', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
model.compile(optimizer=Adam(lr=3e-5),loss=tf.keras.losses.binary_crossentropy, metrics=tf.keras.metrics.Precision())

# Training model...
history = model.fit([train_id,train_mask], y_train, batch_size=32, epochs=5, callbacks=checkpoint, validation_split=0.1)

# Loading model...
model.load_weights('model_bert1.h5')
y_pred = model.predict([test_id, test_mask])

y_int = np.zeros_like(y_pred)
y_int[y_pred > 0.5] = 1

print('Classification report {}'.format(classification_report(y, y_int, zero_division=0)))
print('Confusion matrix {}'.format(multilabel_confusion_matrix(y, y_int)))
print('Accuracy is {}'.format(accuracy_score(y,y_int)))
print('Roc-auc score is {}'.format(roc_auc_score(y, y_pred))) 
