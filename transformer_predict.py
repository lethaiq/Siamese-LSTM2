from time import time
import pandas as pd

import matplotlib
import numpy as np
import tensorflow_hub as hub
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import keras.backend as K

import tensorflow as tf

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist

import pickle

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# File paths
TRAIN_CSV = '../quora/data/train.csv'


# Load training set
train_df = pd.read_csv(TRAIN_CSV)
train_df = train_df.fillna('')
for q in ['question1', 'question2']:
	train_df[q + '_n'] = train_df[q]

X = train_df[['question1_n', 'question2_n']]
Y = train_df['is_duplicate']

# Split to train validation
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size


X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2, random_state=22)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_validation, Y_validation, test_size=0.5, random_state=22)

# X_train = split_and_zero_padding(X_train, max_seq_length)
# X_validation = split_and_zero_padding(X_validation, max_seq_length)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values
Y_test = Y_test.values


# Make sure everything is ok
assert X_train['question1_n'].shape == X_train['question2_n'].shape
assert len(X_train['question1_n']) == len(Y_train)

def UniversalEmbedding(x):
	return embed(tf.squeeze(tf.cast(x, tf.string)), 
		signature="default", as_dict=True)["default"]

embed = hub.Module(module_url)
messages = tf.placeholder(dtype=tf.string, shape=[None])
output = embed(messages)

with tf.Session() as session:
	K.set_session(session)
	session.run(tf.global_variables_initializer())
	session.run(tf.tables_initializer())
	
	# X_test_embed = {}
	# X_test_embed['left'] = []
	# X_test_embed['right'] = []
	# for i in range(0, len(X_test), 1024):
	# 	# print(X_train['question2_n'][i:i+1024])
	# 	x_left = session.run(output, {messages: X_test['question1_n'][i:i+1024]})
	# 	x_right = session.run(output, {messages: X_test['question2_n'][i:i+1024]})
	# 	X_test_embed['left'].append(x_left)
	# 	X_test_embed['right'].append(x_right)
	# 	print(i)

	# pickle.dump(X_test_embed, open('./data/X_test_use.pkl','wb'))
	# print('done')

	# # X_train_embed = {}
	# # X_train_embed['left'] = []
	# # X_train_embed['right'] = []
	# # for i in range(0, len(X_train), 1024):
	# # 	# print(X_train['question2_n'][i:i+1024])
	# # 	x_left = session.run(output, {messages: X_train['question1_n'][i:i+1024]})
	# # 	x_right = session.run(output, {messages: X_train['question2_n'][i:i+1024]})
	# # 	X_train_embed['left'].append(x_left)
	# # 	X_train_embed['right'].append(x_right)
	# # 	print(i)

	# # pickle.dump(X_train_embed, open('./data/X_train_use.pkl','wb'))
	# # print('done')

	# X_valid_embed = {}
	# X_valid_embed['left'] = []
	# X_valid_embed['right'] = []
	# for i in range(0, len(X_validation), 1024):
	# 	x_left = session.run(output, {messages: X_validation['question1_n'][i:i+1024]})
	# 	x_right = session.run(output, {messages: X_validation['question2_n'][i:i+1024]})
	# 	X_valid_embed['left'].append(x_left)
	# 	X_valid_embed['right'].append(x_right)
	# 	print(i)

	# pickle.dump(X_valid_embed, open('./data/X_valid_use.pkl','wb'))
	# print('done')

X_train = pickle.load(open('./data/X_train_use.pkl', 'rb'))
X_train['left'] = np.expand_dims(np.concatenate(X_train['left'], axis=0), 2)
X_train['right'] = np.expand_dims(np.concatenate(X_train['right'], axis=0), 2)

# X_train['left'] = np.concatenate(X_train['left'], axis=0)
# X_train['right'] = np.concatenate(X_train['right'], axis=0)

X_validation = pickle.load(open('./data/X_valid_use.pkl', 'rb'))
X_validation['left'] = np.expand_dims(np.concatenate(X_validation['left'], axis=0), 2)
X_validation['right'] = np.expand_dims(np.concatenate(X_validation['right'], axis=0), 2)

# X_validation['left'] = np.concatenate(X_validation['left'], axis=0)
# X_validation['right'] = np.concatenate(X_validation['right'], axis=0)

X_test = pickle.load(open('./data/X_test_use.pkl', 'rb'))
X_test['left'] = np.expand_dims(np.concatenate(X_test['left'], axis=0), 2)
X_test['right'] = np.expand_dims(np.concatenate(X_test['right'], axis=0), 2)

# X_test['left'] = np.concatenate(X_test['left'], axis=0)
# X_test['right'] = np.concatenate(X_test['right'], axis=0)

X_train = np.array([np.concatenate((X_train['left'][i], X_train['right'][i])) for i in range(len(X_train['left']))])
X_validation = np.array([np.concatenate((X_validation['left'][i], X_validation['right'][i])) for i in range(len(X_validation['left']))])
X_test = np.array([np.concatenate((X_test['left'][i], X_test['right'][i])) for i in range(len(X_test['left']))])

model = tf.keras.models.load_model('./data/SiameseLSTM_USE_fcn.h5')
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