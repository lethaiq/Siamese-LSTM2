from time import time
import pandas as pd

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import tensorflow as tf

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist

import pickle


# File paths
TRAIN_CSV = '../quora/data/train.csv'

# Load training set
train_df = pd.read_csv(TRAIN_CSV)
for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
use_w2v = True

# train_df, embeddings = make_w2v_embeddings(train_df, embedding_dim=embedding_dim, empty_w2v=not use_w2v)
# pickle.dump([train_df, embeddings], open('../quora/data/embeddings_glove.pkl','wb'))
train_df, embeddings = pickle.load(open('../quora/data/embeddings_glove.pkl','rb'))

# Split to train validation
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size

X = train_df[['question1_n', 'question2_n']]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2, random_state=22)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_validation, Y_validation, test_size=0.5, random_state=22)

print(X_train.shape)
print(X_validation.shape)
print(X_test.shape)

X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)
X_test = split_and_zero_padding(X_test, max_seq_length)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values
Y_test = Y_test.values

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

X_train = np.array([np.concatenate((X_train['left'][i], X_train['right'][i])) for i in range(len(X_train['left']))])
X_validation = np.array([np.concatenate((X_validation['left'][i], X_validation['right'][i])) for i in range(len(X_validation['left']))])
X_test = np.array([np.concatenate((X_test['left'][i], X_test['right'][i])) for i in range(len(X_test['left']))])
# --

model = tf.keras.models.load_model('./data/LSTM_glove.h5')
model.summary()

prediction = model.predict(X_test, verbose=1, batch_size=128)
mse = mean_squared_error(Y_test, prediction)
prediction_int = prediction >= 0.5
prediction_int = np.array(prediction_int).astype(int)
acc = accuracy_score(Y_test, prediction_int, normalize=True)
f1 = f1_score(Y_test, prediction_int, average='weighted') 
print(mse, acc)
print(f1)

# prediction = model.predict(X_train, verbose=1, batch_size=512)
# mse = mean_squared_error(Y_train, prediction)
# prediction_int = prediction >= 0.5
# prediction_int = np.array(prediction_int).astype(int)
# acc = accuracy_score(Y_train, prediction_int, normalize=True)
# f1 = f1_score(Y_train, prediction_int, average='weighted') 
# print(mse, acc)
# print(f1)