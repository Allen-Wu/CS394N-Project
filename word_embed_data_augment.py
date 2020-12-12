import numpy as np
import csv
import transformers
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

import model

csv_file_w_data_augment = 'dataset1_train_copy.csv'

words_list = []
words_dict = {}
unique_word_cnt = 0

org_text_lists = []
org_type_idx_list = []
aug_text_lists = []
aug_type_idx_list = []

org_data = []
aug_data = []

num_orig_row = 4879

typename_to_idx = {
    'INTJ': 0,
    'INTP': 1,
    'ENTJ': 2,
    'ENTP': 3,
    'INFJ': 4,
    'INFP': 5,
    'ENFJ': 6,
    'ENFP': 7,
    'ISTJ': 8,
    'ISFJ': 9,
    'ESTJ': 10,
    'ESFJ': 11,
    'ISTP': 12,
    'ISFP': 13,
    'ESTP': 14,
    'ESFP': 15
}

typeidx_to_name = {v: k for k, v in typename_to_idx.items()}

row_cnt = 0
with open(csv_file_w_data_augment) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row_cnt < num_orig_row:
            row = list(map(int, row))
            org_data.append(row)
        else:
            row = list(map(int, row))
            aug_data.append(row)
        row_cnt += 1

org_data = np.array(org_data)
aug_data = np.array(aug_data)

train, test = train_test_split(org_data)
train, val = train_test_split(train)

train = np.append(train, aug_data, 0)

print('Training set shape: {}'.format(str(train.shape)))
print('Validation set shape: {}'.format(str(val.shape)))
print('Test set shape: {}'.format(str(test.shape)))

def idx_to_one_hot_vec(X):
    # X = X.astype('int32')
    vec = np.zeros([len(X), 16])
    for i in range(len(X)):
        vec[i][X[i]] = 1
    # vec = vec.astype('int32')
    return vec

train_input = train[:, :-1]
train_label = train[:, -1]
train_label = idx_to_one_hot_vec(train_label)

val_input = val[:, :-1]
val_label = val[:, -1]
val_label = idx_to_one_hot_vec(val_label)

test_input = test[:, :-1]
test_label = test[:, -1]
test_label = idx_to_one_hot_vec(test_label)

print(train_input.shape)
print(train_label.shape)

batch_size = 4

bert_model = model.create_model(512)

checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_freq='epoch')

bert_model.fit(train_input,
               train_label,
               validation_data=(val_input, val_label),
               verbose=1,
               epochs=20,
               batch_size=batch_size,
               callbacks=[tf.keras.callbacks.EarlyStopping(patience=5),
                          model_checkpoint_callback])
