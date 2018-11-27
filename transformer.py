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
from sklearn.metrics import f1_score

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist
import pickle

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

def UniversalEmbedding(x):
	return embed(tf.squeeze(tf.cast(x, tf.string)), 
		signature="default", as_dict=True)["default"]

# embed = hub.Module(module_url)
# messages = tf.placeholder(dtype=tf.string, shape=[None])
# output = embed(messages)

# with tf.Session() as session:
# 	K.set_session(session)
# 	session.run(tf.global_variables_initializer())
# 	session.run(tf.tables_initializer())
	
# 	X_train_embed = []
# 	for i in range(0, len(X_train), 1024):
# 		x = session.run(output, {messages: X_train[i:i+1024]})
# 		X_train_embed.append(x)
# 		print(i)

# 	pickle.dump(X_train_embed, open('./data/X_train_use.pkl','wb'))
# 	print('done')

# 	X_valid_embed = []
# 	for i in range(0, len(X_validation), 1024):
# 		x = session.run(output, {messages: X_validation[i:i+1024]})
# 		X_valid_embed.append(x)
# 		print(i)

# 	pickle.dump(X_valid_embed, open('./data/X_valid_use.pkl','wb'))
# 	print('done')

X_train = pickle.load(open('./data/X_train_use.pkl', 'rb'))
X_train = np.expand_dims(np.concatenate(X_train, axis=0), 2)
X_validation = pickle.load(open('./data/X_valid_use.pkl', 'rb'))
X_validation = np.expand_dims(np.concatenate(X_validation, axis=0), 2)

#   X_validation_embed = session.run(embed(X_validation))

x = Sequential()
x.add(LSTM(50))
shared_model = x

# The visible layer
left_input = Input(shape=(512,1), dtype='float')
right_input = Input(shape=(512,1), dtype='float')

# Pack it all up into a Manhattan Distance model
malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

print(model.summary())