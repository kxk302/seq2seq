"""
Title: Character-level recurrent sequence-to-sequence model
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2017/09/29
Last modified: 2020/04/26
Description: Character-level recurrent sequence-to-sequence model.
"""

import argparse
import json
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

default_model_parameters = {
    "data_path": "./data/fra.txt",  # Path to the training data file on disk
    "start_sequence_char": "\t",  # Use "tab" as the "start sequence" character
    "end_sequence_char": "\n",  # use "newline" as the "end sequence" character
    "batch_size": 64,  # Batch size for training
    "epochs": 1,  # Number of epochs to train for
    "num_samples": 10000,  # Number of samples to train on
    "validation_split": 0.2,  # Proportion of training data set aside for validation
    "optimizer": "rmsprop",  # Optimizer used in training
    "loss": "categorical_crossentropy",  # Training loss function
    "metrics": ["accuracy"],  # Training metrics
    "latent_dim": 256,  # Latent dimensionality of the encoding space
    "model_name": "./models/seq2seq_plain.keras",  # Name of the file model is saved to
    "model_plot": "./plots/deq2seq_plain.png",  # Name of the model plot file
    "random_seed": 1234,
}


def prepare_training_data(model_parameters):

    training_data = dict()

    # Vectorize the data
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()

    with open(model_parameters["data_path"], "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    for line in lines[: min(model_parameters["num_samples"], len(lines) - 1)]:
        input_text, target_text, _ = line.split("\t")
        # Add start/end sequence chars to target text
        target_text = (
            model_parameters["start_sequence_char"]
            + target_text
            + model_parameters["end_sequence_char"]
        )
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
                # and will not include the start character
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0

    # Reverse-lookup token index to decode sequences back to something readable
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items()
    )

    training_data["encoder_input_data"] = encoder_input_data
    training_data["decoder_input_data"] = decoder_input_data
    training_data["decoder_target_data"] = decoder_target_data
    training_data["input_characters"] = input_characters
    training_data["target_characters"] = target_characters
    training_data["num_encoder_tokens"] = num_encoder_tokens
    training_data["num_decoder_tokens"] = num_decoder_tokens
    training_data["max_encoder_seq_length"] = max_encoder_seq_length
    training_data["max_decoder_seq_length"] = max_decoder_seq_length
    training_data["input_token_index"] = input_token_index
    training_data["target_token_index"] = target_token_index
    training_data["reverse_input_char_index"] = reverse_input_char_index
    training_data["reverse_target_char_index"] = reverse_target_char_index
    training_data["input_texts"] = input_texts

    return training_data


def build_model(model_parameters, num_encoder_tokens, num_decoder_tokens):
    # Define an input sequence and process it
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
    encoder = keras.layers.LSTM(model_parameters["latent_dim"], return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard encoder_outputs and only keep the states
    encoder_states = [state_h, state_c]

    # Set up the decoder, using encoder_states as initial state
    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference
    decoder_lstm = keras.layers.LSTM(
        model_parameters["latent_dim"], return_sequences=True, return_state=True
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn encoder_input_data & decoder_input_data
    # into decoder_target_data
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model


def train_model(model, model_parameters, training_data):
    encoder_input_data = training_data["encoder_input_data"]
    decoder_input_data = training_data["decoder_input_data"]
    decoder_target_data = training_data["decoder_target_data"]

    model.compile(
        optimizer=model_parameters["optimizer"],
        loss=model_parameters["loss"],
        metrics=model_parameters["metrics"],
    )

    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=model_parameters["batch_size"],
        epochs=model_parameters["epochs"],
        validation_split=model_parameters["validation_split"],
    )
    # Save model
    model.save(model_parameters["model_name"])


def restore_model(model, model_parameters):
    # Restore the model and construct the encoder and decoder.
    model = keras.models.load_model(model_parameters["model_name"])

    # Save the model graph to png file
    # tf.keras.utils.plot_model(model, to_file='./plots/plain.png', show_shapes=True, show_dtype=True)

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = keras.Input(shape=(model_parameters["latent_dim"],))
    decoder_state_input_c = keras.Input(shape=(model_parameters["latent_dim"],))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model


def decode_sequence(
    input_seq, encoder_model, decoder_model, model_parameters, training_data
):
    start_sequence_char = model_parameters["start_sequence_char"]
    end_sequence_char = model_parameters["end_sequence_char"]
    num_decoder_tokens = training_data["num_decoder_tokens"]
    reverse_target_char_index = training_data["reverse_target_char_index"]
    max_decoder_seq_length = training_data["max_decoder_seq_length"]
    target_token_index = training_data["target_token_index"]

    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index[start_sequence_char]] = 1.0

    # Sampling loop for a batch of sequences (to simplify, here we assume a batch of size 1)
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (
            sampled_char == end_sequence_char
            or len(decoded_sentence) > max_decoder_seq_length
        ):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]

    return decoded_sentence


def decode_sequences(encoder_model, decoder_model, model_parameters, training_data):
    random_seed = model_parameters["random_seed"]
    input_texts = training_data["input_texts"]
    encoder_input_data = training_data["encoder_input_data"]

    random.seed(model_parameters["random_seed"])
    for idx in range(5):
        # Take one sequence (part of the training set) for trying out decoding
        seq_index = random.randrange(len(input_texts))
        input_seq = encoder_input_data[seq_index : seq_index + 1]
        decoded_sentence = decode_sequence(
            input_seq, encoder_model, decoder_model, model_parameters, training_data
        )
        print("-")
        print("Input sentence:", input_texts[seq_index])
        print("Decoded sentence:", decoded_sentence)


if __name__ == "__main__":
    argParse = argparse.ArgumentParser(
        "To accept configuration file containing parameters for seq2seq model"
    )
    argParse.add_argument("-c", "--config_file", type=str)
    args = argParse.parse_args()

    if args.config_file is not None:
        with open(args.config_file, "r") as fp:
            model_parameters = json.load(fp)
            print(f"Loaded model_parameters: {model_parameters}")
    else:
        model_parameters = default_model_parameters

    training_data = prepare_training_data(model_parameters)
    model = build_model(
        model_parameters,
        training_data["num_encoder_tokens"],
        training_data["num_decoder_tokens"],
    )
    train_model(model, model_parameters, training_data)
    encoder_model, decoder_model = restore_model(model, model_parameters)
    decode_sequences(encoder_model, decoder_model, model_parameters, training_data)
