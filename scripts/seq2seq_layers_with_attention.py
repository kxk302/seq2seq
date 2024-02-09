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
epochs = 100 # Number of epochs to train for.
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
enc_in_1 = keras.Input(shape=(None, num_encoder_tokens), name="encoder_inputs")

# Encoder layer 1
enc1_1 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="enc1_1")
enc1_2 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="enc1_2")

enc1_1_out, enc1_1_h_state, enc1_1_c_state = enc1_1(enc_in_1)
enc1_2_out, enc1_2_h_state, enc1_2_c_state = enc1_2(enc_in_1)

# Encoder layer 2
enc2_1 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="enc2_1")
enc2_2 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="enc2_2")

enc2_1_out, enc2_1_h_state, enc2_1_c_state = enc2_1(enc1_1_out)
enc2_2_out, enc2_2_h_state, enc2_2_c_state = enc2_2(enc1_2_out)

# Decoder inputs
dec_in_1 = keras.Input(shape=(None, num_decoder_tokens), name="dec_in_1")

# Decoder layer 1
dec1_1 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="dec1_1")
dec1_2 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="dec1_2")

dec_out1_1, dec1_1_h_state, dec1_1_c_state = dec1_1(dec_in_1, initial_state=[enc1_1_h_state, enc1_1_c_state])
dec_out1_2, dec1_2_h_state, dec1_2_c_state = dec1_2(dec_in_1, initial_state=[enc1_2_h_state, enc1_2_c_state])

# Decoder layer 2
dec2_1 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="dec2_1")
dec2_2 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="dec2_2")

dec_out2_1, dec2_1_h_state, dec2_1_c_state = dec2_1(dec_out1_1, initial_state=[enc2_1_h_state, enc2_1_c_state])
dec_out2_2, dec2_2_h_state, dec2_2_c_state = dec2_2(dec_out1_2, initial_state=[enc2_2_h_state, enc2_2_c_state])

# Adding attention to the model
attention_1 = keras.layers.Attention(name="attention_1")
attention_2 = keras.layers.Attention(name="attention_2")
attention_outputs_1 = attention_1([dec_out2_1, enc2_1_out])
attention_outputs_2 = attention_2([dec_out2_2, enc2_2_out])

concatenate_1 = keras.layers.Concatenate(axis=-1, name="concatenate_1")
concatenate_2 = keras.layers.Concatenate(axis=-1, name="concatenate_2")
decoder_concat_outputs_1 = concatenate_1([dec_out2_1, attention_outputs_1])
decoder_concat_outputs_2 = concatenate_2([dec_out2_2, attention_outputs_2])

concatenate = keras.layers.Concatenate(axis=-1, name="concatenate")
concatenate_output = concatenate([decoder_concat_outputs_1 , decoder_concat_outputs_2])

dec_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax", name="dec_dense")
dec_out = dec_dense(concatenate_output)

if predict == False:
    model = keras.Model(inputs=[enc_in_1, dec_in_1], outputs=dec_out)
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
    model.save("./models/s2s_layers_with_attention.keras")

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
model = keras.models.load_model("./models/s2s_layers_with_attention.keras")

encoder_inputs = model.input[0]  # input_1
enc1_1_out, enc1_1_state_h, enc1_1_state_c = model.layers[2].output # enc1_1
enc1_2_out, enc1_2_state_h, enc1_2_state_c = model.layers[3].output # enc1_2

enc2_1_out, enc2_1_state_h, enc2_1_state_c = model.layers[5].output # enc_2_1
enc2_2_out, enc2_2_state_h, enc2_2_state_c = model.layers[7].output # enc_2_2  

encoder_outputs = [enc1_1_state_h, enc1_1_state_c, enc1_2_state_h, enc1_2_state_c, enc2_1_state_h, enc2_1_state_c, enc2_2_state_h, enc2_2_state_c, enc2_1_out, enc2_2_out]

encoder_model = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = model.input[1]  # input_2

decoder_state_input_h_1 = keras.Input(shape=(latent_dim,))
decoder_state_input_c_1 = keras.Input(shape=(latent_dim,))
decoder_states_inputs_1 = [decoder_state_input_h_1, decoder_state_input_c_1]
decoder_state_input_h_2 = keras.Input(shape=(latent_dim,))
decoder_state_input_c_2 = keras.Input(shape=(latent_dim,))
decoder_states_inputs_2 = [decoder_state_input_h_2, decoder_state_input_c_2]
decoder_state_input_h_3 = keras.Input(shape=(latent_dim,))
decoder_state_input_c_3 = keras.Input(shape=(latent_dim,))
decoder_states_inputs_3 = [decoder_state_input_h_3, decoder_state_input_c_3]
decoder_state_input_h_4 = keras.Input(shape=(latent_dim,))
decoder_state_input_c_4 = keras.Input(shape=(latent_dim,))
decoder_states_inputs_4 = [decoder_state_input_h_4, decoder_state_input_c_4]

dec1_1 = model.layers[4] # dec1_1
dec1_2 = model.layers[6] # dec1_2

dec1_1_out, dec1_1_state_h, dec1_1_state_c = dec1_1(
    decoder_inputs, initial_state=decoder_states_inputs_1
)
dec1_2_out, dec1_2_state_h, dec1_2_state_c = dec1_2(
    decoder_inputs, initial_state=decoder_states_inputs_2
)

dec2_1 = model.layers[8] # dec2_1
dec2_2 = model.layers[9] # dec2_2

dec2_1_out, dec2_1_state_h, dec2_1_state_c = dec2_1(
    dec1_1_out, initial_state=decoder_states_inputs_3
)
dec2_2_out, dec2_2_state_h, dec2_2_state_c = dec2_2(
    dec1_2_out, initial_state=decoder_states_inputs_4
)

decoder_states = [dec1_1_state_h, dec1_1_state_c, dec1_2_state_h, dec1_2_state_c, dec2_1_state_h, dec2_1_state_c, dec2_2_state_h, dec2_2_state_c]

encoder_outputs_input_1 = keras.Input(shape=(None, latent_dim))
encoder_outputs_input_2 = keras.Input(shape=(None, latent_dim))
r_attention_layer_1 = model.layers[10]
r_attention_layer_2 = model.layers[11]
r_attention_output_1 = r_attention_layer_1([dec2_1_out, enc2_1_out])
r_attention_output_2 = r_attention_layer_2([dec2_2_out, enc2_2_out ])

r_concatenate_layer_1 = model.layers[12]
r_concatenate_layer_2 = model.layers[13]
r_concatenate_output_1 = r_concatenate_layer_1([dec2_1_out, r_attention_output_1])
r_concatenate_output_2 = r_concatenate_layer_2([dec2_2_out, r_attention_output_2])

r_concatenate_layer = model.layers[14]
r_concatenate_output = r_concatenate_layer([r_concatenate_output_1, r_concatenate_output_2])

decoder_dense = model.layers[15] # dense
decoder_outputs = decoder_dense(r_concatenate_output)
decoder_model = keras.Model(
    [decoder_inputs, decoder_state_input_h_1, decoder_state_input_c_1, decoder_state_input_h_2, decoder_state_input_c_2, decoder_state_input_h_3, decoder_state_input_c_3, decoder_state_input_h_4, decoder_state_input_c_4, enc2_1_out, enc2_2_out], [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    h1_1, c1_1, h1_2, c1_2, h2_1, c2_1, h2_2, c2_2, enc2_1_out, enc2_2_out = encoder_model.predict(input_seq)
    states_value = [h1_1, c1_1, h1_2, c1_2, h2_1, c2_1, h2_2, c2_2]
    enc_outputs = [enc2_1_out, enc2_2_out]

    # Generate empty target sequence of length 1.
    target_sequence = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_sequence[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h1_1, c1_1, h1_2, c1_2, h2_1, c2_1, h2_2, c2_2 = decoder_model.predict([target_sequence] + states_value + enc_outputs)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_sequence = np.zeros((1, 1, num_decoder_tokens))
        target_sequence[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h1_1, c1_1, h1_2, c1_2, h2_1, c2_1, h2_2, c2_2]
    return decoded_sentence


"""
You can now generate decoded sentences as such:
"""
random.seed(1234)
for idx in range(5):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    seq_index =random.randrange(len(input_texts))
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)
