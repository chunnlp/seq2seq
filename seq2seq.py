from __future__ import print_function
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys

import argparse
from seq2seq_utils import *

ap = argparse.ArgumentParser()
ap.add_argument('-max_len', type=int, default=200)
ap.add_argument('-vocab_size', type=int, default=10000)
ap.add_argument('-batch_size', type=int, default=64)
ap.add_argument('-layer_num', type=int, default=3)
ap.add_argument('-hidden_dim', type=int, default=256)
ap.add_argument('-nb_epoch', type=int, default=20)
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
    #X = pad_sequences(X, maxlen=X_max_len, dtype='int32')
    #y = pad_sequences(y, maxlen=y_max_len, dtype='int32')
    X = np.array(X)
    y = np.array(y)

    hidden_size = HIDDEN_DIM

    # Creating the network model
    print('[INFO] Compiling model...')
    encoder_inputs = Input(shape=(None, X_vocab_len))
    encoder = LSTM(hidden_size, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, y_vocab_len))
    decoder = LSTM(hidden_size, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder(decoder_inputs,
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
        k_start = 1

        # If any trained weight was found, then load them into the model
        if len(saved_weights) != 0:
            print('[INFO] Saved weights found, loading...')
            epoch = saved_weights[saved_weights.rfind('_')+1:saved_weights.rfind('.')]
            model.load_weights(saved_weights)
            k_start = int(epoch) + 1

        i_end = 0
        for k in range(k_start, NB_EPOCH*100+1):
            # Shuffling the training data every epoch to avoid local minima
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices][:2000]
            y = y[indices][:2000]

            X_sequences_input, y_sequences_input, y_sequences_target = process_data(X, y,
                                                                    X_max_len, y_max_len,
                                                                    X_word_to_ix, y_word_to_ix)

            print('[INFO] Training model: epoch {}th {} samples'.format(k, len(X)))
            model.fit([X_sequences_input, y_sequences_input],
                      y_sequences_target,
                      batch_size=BATCH_SIZE, epochs=1, verbose=2,
                      validation_split=0.2)
            if k % 100 == 0:
                model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))

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
        decoder_outputs, state_h, state_c = decoder(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        X_sequences_input, _, _ = process_data(X, y,
                                               X_max_len, y_max_len,
                                               X_word_to_ix, y_word_to_ix)

        max_target_sequence_length = 10
        for ind in range(100):
            input_sequence = X_sequences_input[ind:ind+1]
            state_value = encoder_model.predict(input_sequence)
            target_sequence = np.zeros((1, 1, y_vocab_len))
            target_sequence[0, 0, 0] = 1.
            stop_condition = False
            target_sentence = ''
            while not stop_condition:
                outputs, h, c = decoder_model.predict(
                    [target_sequence] + state_value)
                sampled_token_index = np.argmax(outputs[0, -1, :])
                output_word = y_ix_to_word[sampled_token_index]
                target_sentence += output_word
                if len(target_sentence) > max_target_sequence_length:
                    stop_condition = True

                target_sequence = np.zeros((1, 1, y_vocab_len))
                target_sequence[0, 0, sampled_token_index] = 1.

                states_values = [h, c]
            print(target_sentence)
            #predicted_sentence = decode_sentence(input_sequence)
            #print(input_sentence)
            #print(predicted_sentence)

