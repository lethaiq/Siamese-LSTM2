import tensorflow as tf
import tensorflow_hub as hub
from time import time
import pandas as pd

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist
import pickle
import keras

from keras import layers
from keras.models import Model
import keras.backend as K

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

X_train = ["{} {}".format(i) for i in X_train.get_values()]
X_validation = ["{} {}".format(i) for i in X_validation.get_values()]

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

embed = hub.Module(module_url)

def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), 
        signature="default", as_dict=True)["default"]