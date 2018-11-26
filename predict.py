import pandas as pd

import tensorflow as tf

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist


# File paths
TRAIN_CSV = './data/train.csv'

# Load training set
train_df = pd.read_csv(TRAIN_CSV)
for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
use_w2v = True

train_df, embeddings = make_w2v_embeddings(train_df, embedding_dim=embedding_dim, empty_w2v=not use_w2v)

# Split to train validation
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size

X = train_df[['question1_n', 'question2_n']]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

# --

model = tf.keras.models.load_model('./data/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
model.summary()

prediction = model.predict([X_validation['left'], X_validation['right']])
print(prediction)
