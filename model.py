import numpy as np
import csv
import transformers
import tensorflow as tf

# BERT based model
def create_model(max_len):
    # maxlen = 512
    input_word_ids = tf.keras.layers.Input(shape=(max_len), dtype=tf.int32,
                                           name="input_word_ids")
    bert_layer = transformers.TFBertModel.from_pretrained('bert-large-uncased')
    bert_outputs = bert_layer(input_word_ids)[0]
    pred = tf.keras.layers.Dense(
        16, activation='softmax')(bert_outputs[:, 0, :])

    model = tf.keras.models.Model(inputs=input_word_ids, outputs=pred)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  metrics=['accuracy'])
    return model

