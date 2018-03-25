import glob
from music21 import converter, instrument, note, chord, stream
import numpy as np
import pickle
import re

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def weight_generator(midi_files, sequence_length, epochs, batch_size, input_notes_file, weights_file, midi_quantity, timestamp,output_name): # weights file should be None for training
    assert weights_file == None
    notes = convert_midis_to_notes(midi_files, input_notes_file, midi_quantity)
    network_input, network_output, n_patterns, n_vocab, pitchnames = prepare_sequences(notes, sequence_length)
    network_input_r, network_output_r = reshape_for_training(network_input, network_output,n_patterns, sequence_length)

    # initializing LSTM model for training
    # weights file initially set to None, but will be passed in full execution function
    model = create_network(network_input_r, n_vocab, weights_file)
    model, weights_file = train_model(model, network_input_r, network_output_r, epochs, batch_size, timestamp, output_name, sequence_length)
    return notes, weights_file

def convert_midis_to_notes(midi_files, input_notes_file, midi_quantity):
    # convert midi file dataset to notes
    notes = [] # list of notes and chords
    note_count = 0

    print("\n**Loading Midi files**")
    for file in glob.glob(midi_files): # loading midi filepaths
        print(file)
        try:
            midi = converter.parse(file) # midi type music21.stream.Score
            parts = instrument.partitionByInstrument(midi)

            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
            # notes_to_parse type music21.stream.iterator.RecursiveIterator
            for e in notes_to_parse:
                if isinstance(e, note.Note):
                    notes.append(str(e.pitch))
                elif isinstance(e, chord.Chord):
                    to_append = '.'.join(str(n) for n in e.normalOrder)
                    notes.append(to_append)
            note_count +=1
        except Exception as e:
            print(e)
            pass
        if note_count >= midi_quantity:
            break
    n_vocab = len(set(notes))
    print("Loaded {} midi files {} notes and {} unique notes".format(note_count, len(notes), n_vocab))
    with open(input_notes_file, 'wb') as f:
        pickle.dump(notes, f)
    print("Input notes/chords stored as {} then pickled at {}".format(type(notes), input_notes_file))
    print("First 20 notes/chords: {}".format(notes[:20]))
    return notes


def prepare_sequences(notes, sequence_length):
    print("\n**Preparing sequences for training**")
    pitchnames = sorted(set(i for i in notes)) # list of unique chords and notes
    n_vocab = len(pitchnames)
    print("Pitchnames (unique notes/chords from 'notes') at length {}: {}".format(len(pitchnames),pitchnames))
    # enumerate pitchnames into dictionary embedding
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    print("Note to integer embedding created at length {}".format(len(note_to_int)))

    network_input = []
    network_output = []

    # i equals total notes less declared sequence length of LSTM (ie 5000 - 100)
    # sequence input for each i is list of notes i to end of sequence length (ie 0-100 for i = 0)
    # sequence output for each i is single note at i + sequence length (ie 100 for i = 0)
    for i in range(0, len(notes) - sequence_length,1):
        sequence_in = notes[i:i + sequence_length] # 100
        sequence_out = notes[i + sequence_length] # 1

        # enumerate notes and chord sequences with note_to_int enumerated encoding
        # network input/output is a list of encoded notes and chords based on note_to_int encoding
        # if 100 unique notes/chords, the encoding will be between 0-100
        input_add = [note_to_int[char] for char in sequence_in]
        network_input.append(input_add) # sequence length
        output_add = note_to_int[sequence_out]
        network_output.append(output_add) # single note

    print("Network input and output created with (pre-transform) lengths {} and {}".format(len(network_input),len(network_output)))
    print("Network input and output first list items: {} and {}".format(network_input[0],network_output[0]))
    print("Network input list item length: {}".format(len(network_input[0])))
    n_patterns = len(network_input) # notes less sequence length
    print("Lengths. N Vocab: {} N Patterns: {} Pitchnames: {}".format(n_vocab,n_patterns, len(pitchnames)))
    return network_input, network_output, n_patterns, n_vocab, pitchnames


def reshape_for_training(network_input, network_output,n_patterns, sequence_length):
    print("\n**Reshaping for training**")
    # convert network input/output from lists to numpy arrays
    # reshape input to (notes less sequence length, sequence length)
    # reshape output to (notes less sequence length, unique notes/chords)
    network_input_r = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_output_r = np_utils.to_categorical(network_output)

    print("Reshaping network input to (notes - sequence length, sequence length) {}".format(network_input_r.shape))
    print("Reshaping network output to (notes - sequence length, unique notes) {}".format(network_output_r.shape))
    return network_input_r, network_output_r


def create_network(network_input, n_vocab, weights_file):
    print("\n**LSTM model initializing**")
    # three layer model
    model = Sequential()
    timesteps = network_input.shape[1]
    data_dim = network_input.shape[2]
    first_layer = 512
    drop = 0.3
    model.add(LSTM(first_layer, input_shape=(timesteps, data_dim), return_sequences=True))
    model.add(Dropout(drop))
    model.add(LSTM(first_layer, return_sequences=True))
    model.add(Dropout(drop))
    model.add(LSTM(first_layer))
    model.add(Dense(first_layer//2))
    model.add(Dropout(drop))
    model.add(Dense(n_vocab)) # based on number of unique system outputs
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

    if weights_file:
        model.load_weights(weights_file)
        print("LSTM model initialized for midi CREATION with weights from {}".format(weights_file))
    else:
        print("LSTM model initialized for TRAINING - weights being generated (no weights file)")
    return model


def train_model(model, network_input_r, network_output_r, epochs, batch_size, timestamp, output_name, sequence_length):
    # saves weights after each epoch
    check_stats = '{epoch:02d}-{loss:.4f}'
    weights_filepath = '../output/{}-weights-'.format(timestamp)
    weights_end = '.hdf5'
    weights_checkpoint = weights_filepath + check_stats + weights_end
    checkpoint = ModelCheckpoint(weights_checkpoint, monitor='loss',verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    print("Fitting Model. \nNetwork Input Shape: {} Network Output Shape: {}".format(network_input_r.shape,network_output_r.shape))
    print("Epochs: {} Batch Size: {}".format(epochs, batch_size))
    model.fit(network_input_r, network_output_r, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

    # saves weights upon training completion
    weights_file = '../output/{}-{}-{}-{}-lstm_weights.hdf5'.format(timestamp, output_name, epochs, sequence_length)
    model.save_weights(weights_file)
    print("TRAINING complete - weights saved at: {}".format(weights_file))
    return model, weights_file
