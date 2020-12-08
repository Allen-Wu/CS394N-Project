import csv
import numpy as np
import tensorflow as tf

input_reddit_vec = []
input_texts = []

# Count vocabulary size
# Map from raw input integer into vocabulary index
reddit_cnt_voc = {}
embed_text_voc = {}

with open('datasets_preprocessed2.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        reddit = row[0][1:-1].replace("'", "").strip().split(',')
        reddit = list(map(int, reddit))
        for x in reddit:
            if x not in reddit_cnt_voc:
                reddit_cnt_voc[x] = len(reddit_cnt_voc)

        embed_text = row[1][1:-1].replace("'", "").strip().split(',')
        embed_text = list(map(int, embed_text))
        for w in embed_text:
            if w not in embed_text_voc:
                embed_text_voc[w] = len(embed_text_voc)
        
        input_reddit_vec.append(reddit)
        input_texts.append(embed_text)

# Map word-count to its index
for i in range(len(input_reddit_vec)):
    for j in range(len(input_reddit_vec[0])):
        input_reddit_vec[i][j] = reddit_cnt_voc[input_reddit_vec[i][j]]

# Map word-embeded id to its index
for i in range(len(input_texts)):
    for j in range(len(input_texts[0])):
        input_texts[i][j] = embed_text_voc[input_texts[i][j]]

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_reddit_cnt_voc = dict((i, char)
                              for char, i in reddit_cnt_voc.items())
reverse_embed_text_voc = dict((i, char)
                              for char, i in embed_text_voc.items())

reddit_vec_len = len(input_reddit_vec[0])
embed_text_len = len(input_texts[0])

reddit_cnt_voc_size = len(reddit_cnt_voc)
embed_text_voc_size = len(embed_text_voc)

print(reddit_cnt_voc_size)
print(embed_text_voc_size)

print("Number of samples:", len(input_reddit_vec))
print("Number of unique subreddit tokens:", reddit_cnt_voc_size)
print("Number of unique text tokens:", embed_text_voc_size)
print("Sequence length for subreddit inputs:", reddit_vec_len)
print("Sequence length for output text:", embed_text_len)

# Encoder input is subreddit vector
# encoder_input_data = np.zeros(
#     (len(input_reddit_vec), reddit_vec_len, reddit_cnt_voc_size), dtype="float16"
# )
#encoder_input_data = np.array(input_reddit_vec, dtype='float16')
# encoder_input_data = tf.convert_to_tensor(encoder_input_data)
# Decoder output is embeded text
# decoder_input_data = np.zeros(
#     (len(input_texts), embed_text_len, embed_text_voc_size), dtype="float16"
# )
#decoder_input_data = np.array(input_texts, dtype='float16')
# decoder_input_data = tf.convert_to_tensor(decoder_input_data)
# for i, (reddit_vec, embed_text_vec) in enumerate(zip(input_reddit_vec, input_texts)):
#     for t, char in enumerate(reddit_vec):
#         encoder_input_data[i, t, reddit_cnt_voc[char]] = 1.0

#     for t, char in enumerate(embed_text_vec):
#         decoder_input_data[i, t, embed_text_voc[char]] = 1.0

# decoder_target_data = np.zeros(
#     (len(input_texts), embed_text_len, embed_text_voc_size), dtype="float32"
# )

# for i, embed_text_vec in enumerate(input_texts):
#     for t, char in enumerate(embed_text_vec):
#         decoder_target_data[i, t, char] = 1.0

# Represent the decoder_target_data in a sparse tensor format
# decoder_target_data_idx = []
# decoder_target_data_val = []
# for i, embed_text_vec in enumerate(input_texts):
#     for t, char in enumerate(embed_text_vec):
#         decoder_target_data_idx.append([i, t, char])
#         decoder_target_data_val.append(char)

# decoder_target_data = tf.sparse.SparseTensor(indices=decoder_target_data_idx,
#                                              values=decoder_target_data_val,
#                                              dense_shape=[len(input_texts), embed_text_len, embed_text_voc_size])

def build_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
    # Define an input sequence and process it.
    # encoder_inputs = tf.keras.Input(shape=(None, num_encoder_tokens))
    e_i = tf.keras.Input(shape=(reddit_vec_len))
    encoder_inputs = tf.keras.layers.Embedding(
        num_encoder_tokens, num_encoder_tokens, input_length=reddit_vec_len)(e_i)

    encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    # decoder_inputs = tf.keras.Input(shape=(None, num_decoder_tokens))
    d_i = tf.keras.Input(shape=(embed_text_len))
    decoder_inputs = tf.keras.layers.Embedding(
        num_decoder_tokens, num_decoder_tokens, input_length=embed_text_len)(d_i)

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = tf.keras.layers.LSTM(
        latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(
        decoder_inputs, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = tf.keras.Model([e_i, d_i], decoder_outputs)
    print(model.summary())

    return model

def my_input_fn(file_path, batch_size, perform_shuffle=False, repeat_count=1):
   def decode_csv(line):
    all_floats = [float()]*8386
    parsed_line = tf.io.decode_csv(line, record_defaults=all_floats)
    # string1 = parsed_line[0].numpy()
    # string2 = parsed_line[1].numpy()
    # reddit = string1[1:-1].replace("'", "").strip().split(',')
    # reddit = list(map(int, reddit))
    # embed_text = string2[1:-1].replace("'", "").strip().split(',')
    # embed_text = list(map(int, embed_text))
    encoder_input_data = parsed_line[0:7874]
    decoder_input_data = parsed_line[7874:8386]
    print(type(encoder_input_data))
    print(type(decoder_input_data))
    feature_names = ['input_1', 'input_2']
    features = []
    features.append(encoder_input_data)
    features.append(decoder_input_data)
    feature_dict = dict(zip(feature_names, features))
    return feature_dict, decoder_input_data

   dataset = (tf.data.TextLineDataset(file_path) # Read text file
       .map(decode_csv)) # Transform each elem by applying decode_csv fn
   if perform_shuffle:
       # Randomizes input using a window of 256 elements (read into memory)
       dataset = dataset.shuffle(buffer_size=32*batch_size)
   dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
   dataset = dataset.batch(batch_size)  # Batch size to use
   dataset = dataset.prefetch(batch_size)
   iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
   batch_features, batch_labels = iterator.get_next()
   return batch_features, batch_labels

    
def dataset_generator():
    encoder_input_data = tf.convert_to_tensor(encoder_input_data)
    decoder_input_data = tf.convert_to_tensor(decoder_input_data)
    inputs = (encoder_input_data, decoder_input_data)
    outputs = tf.sparse.to_dense(decoder_target_data)
    return inputs, outputs

def train_model():
    batch_size = 16
    epochs = 20

    model = build_model(reddit_cnt_voc_size,
                        embed_text_voc_size,
                        256)
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    input_file_path ='/work/06885/zc3968/frontera/NNProject/CS394N-Project/preprocess/new_datasets2.csv'

    checkpoint_filepath = '/work/06885/zc3968/frontera/NNProject/CS394N-Project/estimator/checkpoint'

    estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=checkpoint_filepath)

    estimator.train(input_fn=lambda: my_input_fn(input_file_path, batch_size, perform_shuffle=True), steps=320)

    eval_result = estimator.evaluate(input_fn=lambda: my_input_fn(input_file_path, batch_size, perform_shuffle=True), steps=16)

    print('Eval result: {}'.format(eval_result))


def build_inference_model():
    # Define sampling models
    # Restore the model and construct the encoder and decoder.
    model = tf.keras.models.load_model("s2s")

    # Encoder
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

    # Decoder
    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = tf.keras.Input(shape=(256,), name="input_3")
    decoder_state_input_c = tf.keras.Input(shape=(256,), name="input_4")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_embed = model.layers[3]
    decoder_lstm = model.layers[5]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_embed(decoder_inputs), initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[6]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model


def decode_sequence(input_seq, encoder_model, decoder_model):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, embed_text_voc_size))
    # Populate the first character of target sequence with the start character.
    # TODO: Pick a suitable word to start
    start_token_idx = 100
    target_seq[0, 0, start_token_idx] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_embed_text_voc[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if len(decoded_sentence) > embed_text_len:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, embed_text_voc_size))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]

    return decoded_sentence

# Currently only run training
train_model()