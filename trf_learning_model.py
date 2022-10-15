# import fo utylities
import string
from sympy import im
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import io
import nltk
import json
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, LSTM , Dense,GlobalAveragePooling1D,Flatten, Dropout , GRU
from keras.models import Sequential,load_model
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.api._v2.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.layers import Conv1D, MaxPool1D
from keras.callbacks import TensorBoard, EarlyStopping

#loading databse
with open("data_base.json") as diabetes_dataset:
  dataset = json.load(diabetes_dataset)

#processind and spliting database
def processing_json_dataset(dataset):
    tags = []
    inputs = []
    responses={}
    for intent in dataset['intents']:
        responses[intent['intent']]=intent['answers']
    for lines in intent['questions']:
        inputs.append(lines)
        tags.append(intent['intent'])
    return [tags, inputs, responses]

[tags, inputs, responses] = processing_json_dataset(dataset)
dataset = pd.DataFrame({"inputs":inputs,"tags":tags})
dataset = dataset.sample(frac=1)

dataset['inputs'] = dataset['inputs'].apply(lambda sequence:[ltrs.lower() for ltrs in sequence if ltrs not in string.punctuation])

dataset['inputs'] = dataset['inputs'].apply(lambda wrd: ''.join(wrd))

#tocknization
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(dataset['inputs'])
train = tokenizer.texts_to_sequences(dataset['inputs'])
features = pad_sequences(train)
le = LabelEncoder()
labels = le.fit_transform(dataset['tags'])

input_shape = features.shape[1]
vocabulary = len(tokenizer.word_index)
output_length = le.classes_.shape[0]

#architecture for our chatbot model
m = Sequential()
m.add(Input(shape=(features.shape[1])))
m.add(Embedding(vocabulary + 1,100))
m.add(Conv1D(filters=32,
            kernel_size=5,
            activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_regularizer=tf.keras.regularizers.L2(0.0001),
            kernel_regularizer=tf.keras.regularizers.L2(0.0001),
            activity_regularizer = tf.keras.regularizers.L2(0.0001))) 
m.add(Dropout(0.3))
m.add(LSTM(32, dropout=0.3,return_sequences=True))
m.add(LSTM(16, dropout=0.3,return_sequences=False))
m.add(Dense(128,activation="relu", activity_regularizer = tf.keras.regularizers.L2(0.0001))) 
m.add(Dropout(0.6))
m.add(Dense(output_length, activation="softmax", activity_regularizer = tf.keras.regularizers.L2(0.0001)))

#downloading the weights for a transfered model on google drive
#https://drive.google.com/file/d/1hWNsTs0I0c4ENhZWmkLawdh_zhmmKttd/view?usp=sharing -O data/numerical_caracheters.6B.100d.txt

#readingding the weights for a transfered model on data folder
glove_dir = "data/glove.6B.100d.txt"
embeddings_index = {}
file_ = open(glove_dir, encoding="utf8")
for line in file_:
    arr = line.split()
    single_word = arr[0]
    w = np.asarray(arr[1:],dtype='float32')
    embeddings_index[single_word] = w
file_.close()

#use the weights a pre-tained model
max_words = vocabulary + 1
word_index = tokenizer.word_index
embedding_matrix = np.zeros((max_words,100)).astype(object)
for word , i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
m.layers[0].set_weights([embedding_matrix])
m.layers[0].trainable = False
m.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

#model training
earlyStopping = EarlyStopping(monitor = 'loss', patience = 400, mode = 'min', restore_best_weights = True)
history_training = m.fit(features,labels,epochs=2000, batch_size=64, callbacks=[ earlyStopping])

#saving the weights  of model trainded
model = m.save('model.h5')
