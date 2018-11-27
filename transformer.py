import tensorflow as tf
import tensorflow_hub as hub
from time import time
import pandas as pd

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout
from keras.layers.core import Reshape

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist
import pickle
import keras

from keras import layers
from keras.models import Model
import keras.backend as K
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# File paths
TRAIN_CSV = '../quora/data/train.csv'


# Load training set
train_df = pd.read_csv(TRAIN_CSV)
for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]

X = train_df[['question1_n', 'question2_n']]
Y = train_df['is_duplicate']

# Split to train validation
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size


X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=22)


X_train = np.array(["{} {}".format(i[0], i[1]) for i in X_train.get_values()])
X_validation = np.array(["{} {}".format(i[0], i[1]) for i in X_validation.get_values()])

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

embed = hub.Module(module_url)

def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), 
        signature="default", as_dict=True)["default"]


with tf.Session() as session:
  K.set_session(session)
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())

  X_train_embed = []
  for i in range(0, len(X_train), 32):
      X_train_embed.append(session.run(embed(X_train[i:i+32])))
      print(len(X_train_embed))
#   X_validation_embed = session.run(embed(X_validation))
  

# x = Sequential()
# x.add(Embedding(len(embeddings), embedding_dim,
#                 weights=[embeddings], input_shape=(max_seq_length,), trainable=False))
# # CNN
# # x.add(Conv1D(250, kernel_size=5, activation='relu'))
# # x.add(GlobalMaxPool1D())
# # x.add(Dense(250, activation='relu'))
# # x.add(Dropout(0.3))
# # x.add(Dense(1, activation='sigmoid'))
# # LSTM
# x.add(LSTM(n_hidden))

# shared_model = x

# # The visible layer
# left_input = Input(shape=(max_seq_length,), dtype='int32')
# right_input = Input(shape=(max_seq_length,), dtype='int32')

# # Pack it all up into a Manhattan Distance model
# malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
# model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

# if gpus >= 2:
#     # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
#     model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)
# model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
# model.summary()
# shared_model.summary()