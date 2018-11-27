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
import keras.backend as K
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
train_df = train_df.fillna('')
for q in ['question1', 'question2']:
	train_df[q + '_n'] = train_df[q]

X = train_df[['question1_n', 'question2_n']]
Y = train_df['is_duplicate']

# Split to train validation
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size


X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=22)

# X_train = split_and_zero_padding(X_train, max_seq_length)
# X_validation = split_and_zero_padding(X_validation, max_seq_length)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Make sure everything is ok
assert X_train['question1_n'].shape == X_train['question2_n'].shape
assert len(X_train['question1_n']) == len(Y_train)

def UniversalEmbedding(x):
	return embed(tf.squeeze(tf.cast(x, tf.string)), 
		signature="default", as_dict=True)["default"]

embed = hub.Module(module_url)
messages = tf.placeholder(dtype=tf.string, shape=[None])
output = embed(messages)

# with tf.Session() as session:
# 	K.set_session(session)
# 	session.run(tf.global_variables_initializer())
# 	session.run(tf.tables_initializer())
	
# 	X_train_embed = {}
# 	X_train_embed['left'] = []
# 	X_train_embed['right'] = []
# 	for i in range(0, len(X_train), 1024):
# 		# print(X_train['question2_n'][i:i+1024])
# 		x_left = session.run(output, {messages: X_train['question1_n'][i:i+1024]})
# 		x_right = session.run(output, {messages: X_train['question2_n'][i:i+1024]})
# 		X_train_embed['left'].append(x_left)
# 		X_train_embed['right'].append(x_right)
# 		print(i)

# 	pickle.dump(X_train_embed, open('./data/X_train_use.pkl','wb'))
# 	print('done')

# 	X_valid_embed = {}
# 	X_valid_embed['left'] = []
# 	X_valid_embed['right'] = []
# 	for i in range(0, len(X_validation), 1024):
# 		x_left = session.run(output, {messages: X_validation['question1_n'][i:i+1024]})
# 		x_right = session.run(output, {messages: X_validation['question2_n'][i:i+1024]})
# 		X_valid_embed['left'].append(x_left)
# 		X_valid_embed['right'].append(x_right)
# 		print(i)

# 	pickle.dump(X_valid_embed, open('./data/X_valid_use.pkl','wb'))
# 	print('done')

X_train = pickle.load(open('./data/X_train_use.pkl', 'rb'))
X_train['left'] = np.expand_dims(np.concatenate(X_train['left'], axis=0), 2)
X_train['right'] = np.expand_dims(np.concatenate(X_train['right'], axis=0), 2)
X_validation = pickle.load(open('./data/X_valid_use.pkl', 'rb'))
X_validation['left'] = np.expand_dims(np.concatenate(X_validation['left'], axis=0), 2)
X_validation['right'] = np.expand_dims(np.concatenate(X_validation['right'], axis=0), 2)


# x = Sequential()
# x.add(Conv1D(512, kernel_size=5, activation='relu'))
# x.add(Conv1D(512, kernel_size=3, activation='relu'))
# x.add(GlobalMaxPool1D())
# # x.add(Dense(250, activation='relu'))
# # x.add(Dropout(0.3))
# # x.add(Dense(1, activation='sigmoid'))
# # x.add(LSTM(50))
# # x.add(Dense(250))
# shared_model = x

# The visible layer
# left_input = Input(shape=(512,1), dtype='float')
# right_input = Input(shape=(512,1), dtype='float')

# Pack it all up into a Manhattan Distance model
# malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
# model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
model = Sequential()
model.add(Conv1D(512, kernel_size=5, activation='relu'))
model.add(GlobalMaxPool1D())
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

X_train = np.array([np.concatenate((X_train['left'][i], X_train['right'][i])) for i in range(len(X_train['left']))])
X_validation = np.array([np.concatenate((X_validation['left'][i], X_validation['right'][i])) for i in range(len(X_validation['left']))])

print(X_train.shape)

# Start trainings
training_start_time = time()
callbacks = [EarlyStopping(monitor='val_loss', patience=4)]
# malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
#                            batch_size=1024, epochs=5,
#                            validation_data=([X_validation['left'], X_validation['right']], Y_validation, ), callbacks=callbacks)
malstm_trained = model.fit(X_train, batch_size=1024, epochs=50, validation_data=(X_validation, Y_validation), callbacks=callbacks)
training_end_time = time()
print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                        training_end_time - training_start_time))

model.save('./data/SiameseLSTM_use.h5')

print(str(malstm_trained.history['val_acc'][-1])[:6] +
      "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
print("Done.")
