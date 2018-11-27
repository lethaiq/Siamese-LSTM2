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

input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = layers.Lambda(UniversalEmbedding, output_shape=(512,))(input_text)
dense = layers.Dense(256, activation='relu')(embedding)
lstm = LSTM(500)(Reshape((embedding.shape.as_list()[0], 512, 1))(embedding))
pred = layers.Dense(1, activation='softmax')(lstm)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='binary_crossentropy', 
	optimizer='adam', metrics=['accuracy'])

print(model.summary())

with tf.Session() as session:
  K.set_session(session)
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  history = model.fit(X_train, Y_train,
            validation_data=(X_validation, Y_validation),
            epochs=10,
            batch_size=64)
  model.save('./data/SiameseLSTM_use.h5')