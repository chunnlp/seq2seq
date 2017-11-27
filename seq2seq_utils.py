from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential, Model
from keras.layers import Input, Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop
from nltk import FreqDist
import numpy as np
import os
import datetime

def load_data(source, dist, max_len, vocab_size):

    # Reading raw text from source and destination files
    f = open(source, 'r')
    X_data = f.read()
    f.close()
    f = open(dist, 'r')
    y_data = f.read()
    f.close()

    # Splitting raw text into array of sequences
    X = [text_to_word_sequence(x)[::-1] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]
    y = [text_to_word_sequence(y) for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]
    for i, _ in enumerate(y):
        y[i].insert(0, 'SOS')
        y[i].insert(-1, 'EOS')
    #print(y)

    # Creating the vocabulary set with the most common words
    dist = FreqDist(np.hstack(X))
    X_vocab = dist.most_common(vocab_size-1)
    dist = FreqDist(np.hstack(y))
    y_vocab = dist.most_common(vocab_size-1)

    # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    X_ix_to_word = [word[0] for word in X_vocab]
    # Adding the word "ZERO" to the beginning of the array
    X_ix_to_word.insert(0, 'ZERO')
    # Adding the word 'UNK' to the end of the array (stands for UNKNOWN words)
    X_ix_to_word.append('UNK')

    # Creating the word-to-index dictionary from the array created above
    X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}

    # Converting each word to its index value
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['UNK']

    y_ix_to_word = [word[0] for word in y_vocab]
    #y_ix_to_word.insert(0, 'ZERO')
    y_ix_to_word.append('UNK')
    y_word_to_ix = {word:ix for ix, word in enumerate(y_ix_to_word)}
    for i, sentence in enumerate(y):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                y[i][j] = y_word_to_ix[word]
            else:
                y[i][j] = y_word_to_ix['UNK']
    return (X, len(X_vocab)+2, X_word_to_ix, X_ix_to_word, y, len(y_vocab)+1, y_word_to_ix, y_ix_to_word)

def load_test_data(source, X_word_to_ix, max_len):
    f = open(source, 'r')
    X_data = f.read()
    f.close()

    X = [text_to_word_sequence(x)[::-1] for x in X_data.split('\n') if len(x) > 0 and len(x) <= max_len]
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['UNK']
    return X

def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_size, num_layers):

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
    return model

def create_infer_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_size, num_layers):
    encoder_inputs = Input(shape=(None, X_vocab_len))
    encoder = LSTM(hidden_size, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(hidden_size,))
    decoder_state_input_c = Input(shape=(hidden_size,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True)
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = Dense(y_vocab_len, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    return decoder_model

def process_data(X_word_sentences, y_word_sentences,
                 X_max_len, y_max_len, X_word_to_ix, y_word_to_ix):
    # Vectorizing each element in each sequence
    encoder_input = np.zeros((len(X_word_sentences), X_max_len, len(X_word_to_ix)))
    sequences_input = np.zeros((len(y_word_sentences), y_max_len, len(y_word_to_ix)))
    sequences_target = np.zeros((len(y_word_sentences), y_max_len, len(y_word_to_ix)))
    for i, (X_sentence, y_sentence) in enumerate(zip(X_word_sentences, y_word_sentences)):
        for j, word in enumerate(X_sentence):
            encoder_input[i, j, word] = 1.
        for j, word in enumerate(y_sentence):
            sequences_input[i, j, word] = 1.
            if j > 0:
                sequences_target[i, j - 1, word] = 1.
    return encoder_input, sequences_input, sequences_target

def find_checkpoint_file(folder):
    checkpoint_file = [f for f in os.listdir(folder) if 'checkpoint' in f]
    if len(checkpoint_file) == 0:
        return []
    modified_time = [os.path.getmtime(f) for f in checkpoint_file]
    return checkpoint_file[np.argmax(modified_time)]
