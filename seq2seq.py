from __future__ import print_function
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys

import argparse
from seq2seq_utils import *

ap = argparse.ArgumentParser()
ap.add_argument('-max_len', type=int, default=200)
ap.add_argument('-vocab_size', type=int, default=20000)
ap.add_argument('-batch_size', type=int, default=64)
ap.add_argument('-layer_num', type=int, default=3)
ap.add_argument('-hidden_dim', type=int, default=256)
ap.add_argument('-nb_epoch', type=int, default=100)
ap.add_argument('-mode', default='train')
args = vars(ap.parse_args())

MAX_LEN = args['max_len']
VOCAB_SIZE = args['vocab_size']
BATCH_SIZE = args['batch_size']
LAYER_NUM = args['layer_num']
HIDDEN_DIM = args['hidden_dim']
NB_EPOCH = args['nb_epoch']
MODE = args['mode']

if __name__ == '__main__':
    # Loading input sequences, output sequences and the necessary mapping dictionaries
    print('[INFO] Loading data...')
    X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word = load_data('source', 'dest', MAX_LEN, VOCAB_SIZE)

    # Finding the length of the longest sequence
    X_max_len = max([len(sentence) for sentence in X])
    y_max_len = max([len(sentence) for sentence in y])

    # Padding zeros to make all sequences have a same length with the longest one
    print('[INFO] Zero padding...')
    X = pad_sequences(X, maxlen=X_max_len, dtype='int32')
    y = pad_sequences(y, maxlen=y_max_len, dtype='int32')

    hidden_size = HIDDEN_DIM

    # Creating the network model
    print('[INFO] Compiling model...')
    encoder_inputs = Input(shape=(None,))
    x = Embedding(X_vocab_len, hidden_size, mask_zero=True)(encoder_inputs)
    encoder = LSTM(hidden_size, return_state=True)
    encoder_outputs, state_h, state_c = encoder(x)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    x = Embedding(y_vocab_len, hidden_size, mask_zero=True)(decoder_inputs)
    decoder = LSTM(hidden_size, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder(x,
                                    initial_state=encoder_states)
    decoder_dense = Dense(y_vocab_len, activation='softmax')
    outputs = decoder_dense(decoder_outputs)
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop')
    #model = create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, HIDDEN_DIM, LAYER_NUM)

    # Finding trained weights of previous epoch if any
    saved_weights = find_checkpoint_file('.')

    # Training only if we chose training mode
    if MODE == 'train':
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        y_sequences_target = process_data(y, y_max_len, y_word_to_ix)

        model.fit([X, y],
                  y_sequences_target,
                  epochs=NB_EPOCH,
                  validation_split=0.2,
                  #steps_per_epoch=len(X_sequences_input) // BATCH_SIZE)
                  batch_size=BATCH_SIZE)
        model.save_weights('checkpoint.hdf5')

    # Performing test if we chose test mode
    else:
        # Only performing test if there is any saved weights
        if len(saved_weights) == 0:
            print("The network hasn't been trained! Program will exit...")
            sys.exit()
        else:
            model.load_weights(saved_weights)
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(hidden_size,))
        decoder_state_input_c = Input(shape=(hidden_size,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        x = Embedding(y_vocab_len, hidden_size)(decoder_inputs)
        decoder_outputs, state_h, state_c = decoder(
            x, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        X_sequences_input, _, _ = process_data(X[:100], y[:100],
                                               X_max_len, y_max_len,
                                               X_word_to_ix, y_word_to_ix)

        max_target_sequence_length = 2 * y_max_len
        for ind in range(100):
            input_sequence = X[ind:ind+1]
            print(input_sequence)
            state_value = encoder_model.predict(input_sequence)
            target_sequence = np.zeros((1, 1))
            target_sequence[0, 0] = y_word_to_ix['SOS']
            stop_condition = False
            target_sentence = []
            while not stop_condition:
                print(target_sequence)
                outputs, h, c = decoder_model.predict(
                    [target_sequence] + state_value)
                sampled_token_index = np.argmax(outputs[0, -1, :])
                output_word = y_ix_to_word[sampled_token_index]
                if output_word == 'EOS' or output_word == 'eos':
                    print(output_word)
                target_sentence.append(output_word)
                if (output_word == 'EOS' or
                    len(target_sentence) > max_target_sequence_length):
                    stop_condition = True

                target_sequence = np.zeros((1, 1))
                target_sequence[0, 0] = sampled_token_index

                states_values = [h, c]
            print(target_sentence)
            #predicted_sentence = decode_sentence(input_sequence)
            #print(input_sentence)
            #print(predicted_sentence)

