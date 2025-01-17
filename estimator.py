import csv
import numpy as np
import tensorflow as tf
from convert_datasets import csv_preprocess
import tqdm

# tf.executing_eagerly()
# tf.config.experimental_run_functions_eagerly(True)

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

old_csv_file = '/home/shiyu/CS394N/CS394N-Project/1.csv'
new_csv_file = '/home/shiyu/CS394N/CS394N-Project/1_new.csv'
checkpoint_filepath = '/home/shiyu/CS394N/CS394N-Project/checkpoint'

reddit_cnt_voc, embed_text_voc, start_token, end_token, reddit_vec_len, embed_text_len = csv_preprocess(
    old_csv_file, new_csv_file)

# Decoder input data: Start token in the beginning, end token in the end
# Decoder target data: Two end tokens in the end
start_token_idx = embed_text_voc[start_token]
end_token_idx = embed_text_voc[end_token]

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_reddit_cnt_voc = dict((i, char)
                              for char, i in reddit_cnt_voc.items())
reverse_embed_text_voc = dict((i, char)
                              for char, i in embed_text_voc.items())

reddit_cnt_voc_size = len(reddit_cnt_voc)
embed_text_voc_size = len(embed_text_voc)

print(reddit_cnt_voc_size)
print(embed_text_voc_size)

# print("Number of samples:", len(input_reddit_vec))
print("Number of unique subreddit tokens:", reddit_cnt_voc_size)
print("Number of unique text tokens:", embed_text_voc_size)
print("Sequence length for subreddit inputs:", reddit_vec_len)
print("Sequence length for output text:", embed_text_len)


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
        all_floats = [float()]*(reddit_vec_len + embed_text_len)
        parsed_line = tf.io.decode_csv(line, record_defaults=all_floats)

        # Transform input token into idx
        for i in range(reddit_vec_len):
            parsed_line[i] = tf.constant(reddit_cnt_voc[int(parsed_line[i].numpy())])
        
        
        # Add start token
        # parsed_line.insert(reddit_vec_len, tf.constant(float(start_token_idx)))
        for i in range(0, embed_text_len):
            parsed_line[reddit_vec_len + i] = tf.constant(
                embed_text_voc[int(parsed_line[reddit_vec_len + i].numpy())])

        
        # Add end token
        # parsed_line.append(tf.constant(float(end_token_idx)))

        # Skip first start token
        decoder_target_data_raw = parsed_line[(reddit_vec_len + 1):]

        decoder_target_data_raw = [x.numpy() for x in decoder_target_data_raw]
        decoder_target_data_raw.append(np.float32(end_token_idx))
        decoder_target_data_raw = np.array(decoder_target_data_raw)

        decoder_target_data = np.zeros((embed_text_len, embed_text_voc_size))

        # Manually embed the decoder_target_data
        for t, char in enumerate(decoder_target_data_raw):
            decoder_target_data[t, int(char)] = 1.0
        
        return parsed_line, decoder_target_data

    def decode_csv_wrapper(line):
        parsed_line, decoder_target_data = tf.py_function(
            func=decode_csv, inp=[line], Tout=[tf.float32, tf.float32])
        feature_names = ['input_1', 'input_2']
        features = []
        encoder_input_data = parsed_line[0:reddit_vec_len]
        encoder_input_data.set_shape([reddit_vec_len])
        decoder_input_data = parsed_line[reddit_vec_len:]
        decoder_input_data.set_shape([embed_text_len])
        features.append(encoder_input_data)
        features.append(decoder_input_data)
        feature_dict = dict(zip(feature_names, features))
        decoder_target_data.set_shape([embed_text_len, embed_text_voc_size])
        return feature_dict, decoder_target_data

    tf.executing_eagerly()
    dataset = (tf.data.TextLineDataset(file_path) # Read text file
               .map(decode_csv_wrapper))  # Transform each elem by applying decode_csv fn
    if perform_shuffle:
        # Randomizes input using a window of 512 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=32*batch_size)
    dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    dataset = dataset.prefetch(batch_size)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def serving_input_receiver_fn():

    input_1 = tf.compat.v1.placeholder(
        dtype=tf.float32, shape=[None, reddit_vec_len], name='input_1')
    input_2 = tf.compat.v1.placeholder(
        dtype=tf.float32, shape=[None, embed_text_len], name='input_2')
    receiver_tensors = {'input_1': input_1, 'input_2': input_2}
    return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)


def train_model():
    batch_size = 1
    
    tf.get_logger().setLevel('INFO')
    model = build_model(reddit_cnt_voc_size,
                        embed_text_voc_size,
                        512)
    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    input_file_path = new_csv_file

    config = tf.estimator.RunConfig().replace(log_step_count_steps=1,
                                              save_checkpoints_steps=10)

    estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                      model_dir=checkpoint_filepath,
                                                      config=config)

    estimator.train(input_fn=lambda: my_input_fn(input_file_path, batch_size, perform_shuffle=True), steps=10)

    eval_result = estimator.evaluate(input_fn=lambda: my_input_fn(input_file_path, batch_size, perform_shuffle=True), steps=2)

    print('Eval result: {}'.format(eval_result))

    estimator.export_saved_model('saved_model', serving_input_receiver_fn)


# Load saved Keras model
def build_inference_model(model_file):
    # Define sampling models
    # Restore the model and construct the encoder and decoder.
    model = build_model(reddit_cnt_voc_size,
                        embed_text_voc_size,
                        512)
    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.load_weights(model_file)
    
    # Encoder
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

    # Decoder
    # decoder_inputs = model.input[1]  # input_2

    decoder_inputs = tf.keras.Input(shape=(1), name='input_3')
    # Embed layer embed_text_voc_size
    decoder_embed = tf.keras.layers.Embedding(
        embed_text_voc_size, embed_text_voc_size, input_length=1)

    decoder_state_input_h = tf.keras.Input(shape=(512,), name="input_4")
    decoder_state_input_c = tf.keras.Input(shape=(512,), name="input_5")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    # decoder_embed = model.layers[3]
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


# Inference based on saved Keras model
def decode_sequence(input_seq, encoder_model, decoder_model):
    # Encode the input as state vectors.
    input_seq = np.reshape(input_seq, (1, len(input_seq)))
    states_value = encoder_model.predict(input_seq)
    last_h = np.array(states_value[0][-1]).reshape((1, len(states_value[0][-1])))
    last_c = np.array(states_value[1][-1]).reshape((1, len(states_value[1][-1])))
    states_value = [last_h, last_c]

    # Generate empty target sequence of length 1.
    target_seq = np.array([float(start_token_idx)])

    decoded_sentence = []
    for i in range(embed_text_len):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        sampled_char = reverse_embed_text_voc[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Update the target sequence (of length 1).
        # target_seq[0][i] = sampled_token_index
        target_seq = np.array([float(sampled_token_index)])

        # Update states
        states_value = [h, c]

    return decoded_sentence


def append_to_csv(csv_file_to_write, transformed_seqs, personality_type):
    # Append to the original dataset
    with open(csv_file_to_write, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        for i in range(len(transformed_seqs)):
            csv_writer.writerow(
                transformed_seqs[i] + [typename_to_idx[personality_type[i]]])


def reddit_vec_to_embed_text(reddit_csv_file, encoder_model, decoder_model, csv_file_to_write):
    personality_type = []
    input_seqs = []
    first_row = True

    append_period = 10

    skip_row_cnt = 0

    with open(reddit_csv_file, "r") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if first_row:
                first_row = False
            else:
                skip_row = False
                personality_type.append(row[0])
                input_seq = row[1:]
                # Transform into idx
                for i in range(len(input_seq)):
                    if int(input_seq[i]) not in reddit_cnt_voc:
                        # Skip this row
                        skip_row_cnt += 1
                        skip_row = True
                        break
                        # raise Exception("The input subreddit count token is not in the training vocabulary")
                        # input_seq[i] = 0
                    else:
                        input_seq[i] = reddit_cnt_voc[int(input_seq[i])]
                if not skip_row:
                    input_seqs.append(input_seq)
    num_reddit_vec = len(input_seqs)
    transformed_seqs = []
    input_seqs = np.array(input_seqs)
    for i in tqdm.tqdm(range(num_reddit_vec)):
        # Append per append_period
        if (i % append_period) == 0:
            append_to_csv(csv_file_to_write, transformed_seqs, personality_type[(i - append_period):i])
            transformed_seqs = []
        transformed_seq = decode_sequence(input_seqs[i], encoder_model, decoder_model)
        # Ignore the last 2 end tokens
        transformed_seqs.append(transformed_seq[:-2])

    # Append the remaining
    append_to_csv(csv_file_to_write,
                  transformed_seqs,
                  personality_type[(len(personality_type) - len(transformed_seqs)):])

    print('# of rows skipped: {}'.format(str(skip_row_cnt)))
    # Append to the original dataset
    # with open(csv_file_to_write, 'a+', newline='') as write_obj:
    #     # Create a writer object from csv module
    #     csv_writer = csv.writer(write_obj)
    #     # Add contents of list as last row in the csv file
    #     for i in range(len(transformed_seqs)):
    #         csv_writer.writerow(transformed_seqs[i] + [typename_to_idx[personality_type[i]]])

# Load saved estimator model
def load_estimator(model_file):
    tf.compat.v1.disable_eager_execution()
    model = tf.saved_model.load(model_file)
    return model.signatures["serving_default"]


# Inference based on saved estimator model
def estimator_inference(input_seq, estimator):
    target_seq = np.ones((1, embed_text_len)) * start_token_idx
    target_seq = tf.reshape(tf.convert_to_tensor(target_seq), [1, embed_text_len])

    estimator = load_estimator('saved_model/1607498132')

    input_seq = []
    with open(new_csv_file, "r") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            input_seq = row[0:reddit_vec_len]
            break
    for i in range(len(input_seq)):
        input_seq[i] = float(reddit_cnt_voc[int(input_seq[i])])
    input_seq = tf.reshape(tf.convert_to_tensor(
        input_seq, dtype=tf.float32), [1, reddit_vec_len])
    # Use estimator to do inference
    # output = estimator_inference(input_seq, estimator)

    target_seq = np.ones((1, embed_text_len)) * start_token_idx
    target_seq = tf.reshape(tf.convert_to_tensor(
        target_seq, dtype=tf.float32), [1, embed_text_len])
    with tf.compat.v1.Session() as sess:
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        prediction_result = estimator(input_1=input_seq, input_2=target_seq)
        output = prediction_result['dense'].eval()
        predict_word_idx = np.argmax(output[0][0])


# Currently only run training
# train_model()
encoder_model, decoder_model = build_inference_model('/home/shiyu/CS394N/CS394N-Project/checkpoint_test/model1/model.ckpt-1')
reddit_vec_to_embed_text(
    '/home/shiyu/CS394N/CS394N-Project/reddit_preprocessed.csv',
    encoder_model,
    decoder_model,
    '/home/shiyu/CS394N/CS394N-Project/dataset1_train.csv')
