import csv
import numpy as np
import tensorflow as tf

input_reddit_vec = []
input_texts = []

# Count vocabulary size
# Map from raw input integer into vocabulary index
reddit_cnt_voc = {}
embed_text_voc = {}

with open('sample.csv') as csvfile:
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
encoder_input_data = np.array(input_reddit_vec, dtype='float16')
# Decoder output is embeded text
# decoder_input_data = np.zeros(
#     (len(input_texts), embed_text_len, embed_text_voc_size), dtype="float16"
# )
decoder_input_data = np.array(input_texts, dtype='float16')

print(encoder_input_data.shape)
print(decoder_input_data.shape)

# for i, (reddit_vec, embed_text_vec) in enumerate(zip(input_reddit_vec, input_texts)):
#     for t, char in enumerate(reddit_vec):
#         encoder_input_data[i, t, reddit_cnt_voc[char]] = 1.0

#     for t, char in enumerate(embed_text_vec):
#         decoder_input_data[i, t, embed_text_voc[char]] = 1.0

decoder_target_data = np.copy(decoder_input_data)

decoder_target_data = np.zeros(
    (len(input_texts), embed_text_len, embed_text_voc_size), dtype="float32"
)

for i, embed_text_vec in enumerate(input_texts):
    for t, char in enumerate(embed_text_vec):
        decoder_target_data[i, t, char] = 1.0

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

    # print(model.summary())

    return model


batch_size = 16
epochs = 20

model = build_model(reddit_cnt_voc_size,
                    embed_text_voc_size,
                    256)
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)

# Save model
model.save("s2s")
