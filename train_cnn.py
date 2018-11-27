from time import time
import pandas as pd

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout, MaxPool1D

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

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=22)

X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

X_train = np.array([np.concatenate((X_train['left'][i], X_train['right'][i])) for i in range(len(X_train['left']))])
X_validation = np.array([np.concatenate((X_validation['left'][i], X_validation['right'][i])) for i in range(len(X_validation['left']))])
# --

print(X_train.shape, Y_train.shape)
print(X_validation.shape, Y_validation.shape)

# Model variables
gpus = 1
batch_size = 1024 * gpus
n_epoch = 100
n_hidden = 50

# Define the shared model
x = Sequential()
x.add(Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(max_seq_length*2,), trainable=False))
x.add(Conv1D(256, kernel_size=5, activation='relu'))
x.add(MaxPool1D(5))
x.add(GlobalMaxPool1D())
x.add(Dense(250, activation='relu'))
x.add(Dropout(0.5))
x.add(Dense(150, activation='relu'))
x.add(Dropout(0.5))
x.add(Dense(1, activation='sigmoid'))
x.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
x.summary()

model = x

try:
    # Start trainings
    training_start_time = time()
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
    malstm_trained = model.fit(X_train, Y_train,
                            batch_size=batch_size, epochs=n_epoch,
                            validation_data=(X_validation, Y_validation), callbacks=callbacks)

    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                            training_end_time - training_start_time))
    model.save('./data/CNN_word2vec.h5')
except KeyboardInterrupt:
    model.save('./data/CNN_word2vec.h5')


print(str(malstm_trained.history['val_acc'][-1])[:6] +
      "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
print("Done.")
