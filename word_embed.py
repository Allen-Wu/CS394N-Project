import numpy as np
import csv
import transformers
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

import model

words_list = []
words_dict = {}
unique_word_cnt = 0
text_lists = []
type_idx_list = []

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

with open('mbti_preprocessed.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    firstLine = True
    for row in readCSV:
        if firstLine:
            firstLine = False
            continue
        words = []
        text = row[1]
        text_lists.append(text)
        type_idx_list.append(typename_to_idx[row[0]])

        sentences = text.split(" ")
        paragraph = ""
        for w in sentences:
            if not w == "":
                words.append(w)
                if not w in words_dict:
                    unique_word_cnt += 1
                    words_dict[w] = 1
                else:
                    words_dict[w] += 1
        words_list.append(words)

print('# of unique words: {}'.format(str(unique_word_cnt)))
assert len(words_dict) == unique_word_cnt

# Show the top10 words
# cnt = 0
# for k, v in sorted(words_dict.items(), key=lambda item: item[1], reverse=True):
#     if cnt > 10:
#         break
#     print('{} occurs {} times'.format(k, str(v)))
#     cnt += 1

# Use sklearn transformer
text_lists = np.array(text_lists)
type_idx_list = np.array(type_idx_list)
type_idx_list = type_idx_list.reshape([len(type_idx_list), 1])

# Word counting
cntizer = CountVectorizer(analyzer="word",
                          max_features=512,
                          tokenizer=None,
                          preprocessor=None,
                          stop_words=None,
                          max_df=0.9,
                          min_df=0.1)

# Learn the vocabulary dictionary and return term-document matrix
X_cnt = cntizer.fit_transform(text_lists)
top_feature_names = cntizer.get_feature_names()


def remove_extra_words(str):
    words = str.split()
    if len(words) <= 512:
        return str
    w = []
    for word in words:
        if word in top_feature_names:
            w.append(word)
    # Truncate to 512
    if len(w) > 512:
        del w[512:]
    return ' '.join(w)


# Use BERT tokenizer to generate input_ids
text_lists = text_lists.tolist()
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
for i in range(len(text_lists)):
    t = remove_extra_words(text_lists[i])
    tokens = tokenizer(t)['input_ids']
    if len(tokens) > 512:
        del tokens[512:]
    elif len(tokens) < 512:
        tokens.extend([0] * (512 - len(tokens)))
    text_lists[i] = tokens

X_tfidf = np.array(text_lists)

# Transform the count matrix to a normalized tf or tf-idf representation
# tfizer = TfidfTransformer()

# Learn the idf vector (fit) and transform a count matrix to a tf-idf representation
# X_tfidf = tfizer.fit_transform(X_cnt).toarray()

all_data = np.append(X_tfidf, type_idx_list, 1)

print('All dataset size: {}'.format(all_data.shape))

train, test = train_test_split(all_data)
train, val = train_test_split(train)

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
