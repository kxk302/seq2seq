"""
Title: Character-level recurrent sequence-to-sequence model
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2017/09/29
Last modified: 2020/04/26
Description: Character-level recurrent sequence-to-sequence model.
Accelerator: GPU
"""
"""
## Introduction

This example demonstrates how to implement a basic character-level
recurrent sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

**Summary of the algorithm**

- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    It uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
"""

"""
## Setup
"""
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

"""
## Download the data
"""

"""shell
!curl -O http://www.manythings.org/anki/fra-eng.zip
!unzip fra-eng.zip
"""

"""
## Configuration
"""

batch_size = 64  # Batch size for training.
epochs = 1  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = "./data/fra.txt"

predict = False
if len(sys.argv) == 2:
    if(sys.argv[1]) == "predict":
        predict = True

"""
## Prepare the data
"""

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

"""
## Build the model
"""

encoder_inputs = keras.Input(shape=(None, num_encoder_tokens), name="encoder_inputs")
encoder_1 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="encoder1")
encoder_2 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="encoder2")

encoder_outputs_1, encoder_h_state_1, encoder_c_state_1 = encoder_1(encoder_inputs)
encoder_outputs_2, encoder_h_state_2, encoder_c_state_2 = encoder_2(encoder_inputs)

decoder_inputs = keras.Input(shape=(None, num_decoder_tokens), name="decoder_inputs")
decoder_lstm_1 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm1")
decoder_lstm_2 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm2")

decoder_outputs_1, _, _ = decoder_lstm_1(decoder_inputs, initial_state=[encoder_h_state_1, encoder_c_state_1])
decoder_outputs_2, _, _ = decoder_lstm_2(decoder_inputs, initial_state=[encoder_h_state_2, encoder_c_state_2])

# Adding attention to the model
attention_1 = keras.layers.Attention(name="attention_1")
attention_2 = keras.layers.Attention(name="attention_2")
attention_outputs_1 = attention_1([decoder_outputs_1, encoder_outputs_1])
attention_outputs_2 = attention_2([decoder_outputs_2, encoder_outputs_2])

concatenate_1 = keras.layers.Concatenate(axis=-1, name="concatenate_1")
concatenate_2 = keras.layers.Concatenate(axis=-1, name="concatenate_2")
decoder_concat_outputs_1 = concatenate_1([decoder_outputs_1, attention_outputs_1])
decoder_concat_outputs_2 = concatenate_2([decoder_outputs_2, attention_outputs_2])

concatenate = keras.layers.Concatenate(axis=-1, name="concatenate")
concatenate_output = concatenate([decoder_concat_outputs_1 , decoder_concat_outputs_2])

decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(concatenate_output)

if predict == False:

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())

    """
    ## Train the model
    """

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
    model.save("s2s_cells_with_attention.keras")

"""
## Run inference (sampling)

1. encode input and retrieve initial decoder state
2. run one step of decoder with this initial state
and a "start of sequence" token as target.
Output will be the next target token.
3. Repeat with the current target token and current states
"""

# Define sampling models
# Restore the model and construct the encoder and decoder.
model = keras.models.load_model("s2s_cells_with_attention.keras")

encoder_inputs = model.input[0]  # input_1
encoder_outputs_1, state_h_enc_1, state_c_enc_1 = model.layers[2].output  # lstm_1
encoder_outputs_2, state_h_enc_2, state_c_enc_2 = model.layers[3].output  # lstm_2
encoder_states = [state_h_enc_1, state_c_enc_1, state_h_enc_2, state_c_enc_2]
encoder_model = keras.Model(encoder_inputs, [encoder_outputs_1, encoder_outputs_2, state_h_enc_1, state_c_enc_1, state_h_enc_2, state_c_enc_2])

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h_1 = keras.Input(shape=(latent_dim,))
decoder_state_input_c_1 = keras.Input(shape=(latent_dim,))
decoder_states_inputs_1 = [decoder_state_input_h_1, decoder_state_input_c_1]
decoder_state_input_h_2 = keras.Input(shape=(latent_dim,))
decoder_state_input_c_2 = keras.Input(shape=(latent_dim,))
decoder_states_inputs_2 = [decoder_state_input_h_2, decoder_state_input_c_2]
decoder_lstm_1 = model.layers[4]
decoder_outputs_1, state_h_dec_1, state_c_dec_1 = decoder_lstm_1(
    decoder_inputs, initial_state=decoder_states_inputs_1
)
decoder_lstm_2 = model.layers[5]
decoder_outputs_2, state_h_dec_2, state_c_dec_2 = decoder_lstm_2(
    decoder_inputs, initial_state=decoder_states_inputs_2
)
decoder_states = [state_h_dec_1, state_c_dec_1, state_h_dec_2, state_c_dec_2]

encoder_outputs_input_1 = keras.Input(shape=(None, latent_dim))
encoder_outputs_input_2 = keras.Input(shape=(None, latent_dim))
r_attention_layer_1 = model.layers[6]
r_attention_layer_2 = model.layers[7]
r_attention_output_1 = r_attention_layer_1([decoder_outputs_1, encoder_outputs_input_1])
r_attention_output_2 = r_attention_layer_2([decoder_outputs_2, encoder_outputs_input_2])

r_concatenate_layer_1 = model.layers[8]
r_concatenate_layer_2 = model.layers[9]
r_concatenate_output_1 = r_concatenate_layer_1([decoder_outputs_1, r_attention_output_1])
r_concatenate_output_2 = r_concatenate_layer_2([decoder_outputs_2, r_attention_output_2])

r_concatenate_layer = model.layers[10]
r_concatenate_output = r_concatenate_layer([r_concatenate_output_1, r_concatenate_output_2])

decoder_dense = model.layers[11]
decoder_outputs = decoder_dense(r_concatenate_output)
decoder_model = keras.Model(
    [decoder_inputs, decoder_state_input_h_1, decoder_state_input_c_1, decoder_state_input_h_2, decoder_state_input_c_2, encoder_outputs_input_1, encoder_outputs_input_2], [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    encoder_outputs_1, encoder_outputs_2, state_h_enc_1, state_c_enc_1, state_h_enc_2, state_c_enc_2 = encoder_model.predict(input_seq)
    states_value = [state_h_enc_1, state_c_enc_1, state_h_enc_2, state_c_enc_2, encoder_outputs_1, encoder_outputs_2]

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h1, c1, h2, c2, encoder_outputs_1, encoder_outputs_2]
    return decoded_sentence


"""
You can now generate decoded sentences as such:
"""
random.seed(1234)
for idx in range(5):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    seq_index = random.randrange(len(input_texts))
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)
