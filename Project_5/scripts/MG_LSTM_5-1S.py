import glob
from music21 import converter, instrument, note, chord, stream
import numpy as np
import pickle
import datetime
import re
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from keras import backend as K

print(sys.executable)
print("Available GPUs: {}".format(K.tensorflow_backend._get_available_gpus()))

# Hyperparameters
input_notes_file = '../data/input_notes'
output_notes_file = '../data/output_notes'
midi_files = '../data/MidiWorld/Pop/AceofBase*.mid'
weights_file = None

sequence_length = 100 # the LSTM RNN will consider this many notes
node1 = 512 # first layer nodes
node2 = 256 # second layer nodes
drop = 0.3 # dropout
epochs = 100
batch_size = 64
notes_generated = 500 # output will contain this many notes
sound = instrument.Piano() # declare a music21 package instrument

output_name = midi_files.split('/')[-2]
timestamp = str(datetime.datetime.now()).split()[0].replace('-','')

# EXECUTE ALL FUNCTIONS
# if midi file provided, will generate midi from midi files (by training weights)
# if notes and weights file provided will generate midi from weights
# def full_execution(midi_files):
#     # if weights_file and notes:
#     #     output_notes = weights_to_midi(notes, weights_file)
#     notes, weights_file = midi_files_to_weights(midi_files)
#     output_notes = weights_to_midi(notes, weights_file)
    # else:
    #     print("Invalid combination.\nConvert midi files to midi with midi files\nor weights to midi with notes and weights")


# input midi files, output weights
def full_execution(midi_files):
    notes = convert_midis_to_notes(midi_files)
    network_input, network_output, n_patterns, n_vocab, pitchnames = prepare_sequences(notes)
    network_input_r, network_output_r = reshape_for_training(network_input, network_output,n_patterns)

    # initializing LSTM model for training
    model = create_network(network_input_r, n_vocab)
    model, weights_file = train_model(model, network_input_r, network_output_r, epochs, batch_size)
    # repeating function so weights to midi can be derived from notes file alone
    # network_input, network_output, n_patterns, n_vocab, pitchnames = prepare_sequences(notes)
    # initialize LSTM for midi creation
    normalized_input = reshape_for_creation(network_input, n_patterns, sequence_length, n_vocab)
    model = create_network(normalized_input, n_vocab, weights_file)
    prediction_output = generate_notes(model, network_input, pitchnames,n_vocab)
    output_notes = create_midi(prediction_output)
    return output_notes


def convert_midis_to_notes(midi_files):
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
    n_vocab = len(set(notes))
    print("Loaded {} midi files {} notes and {} unique notes".format(note_count, len(notes), n_vocab))
    with open(input_notes_file, 'wb') as f:
        pickle.dump(notes, f)
    print("Input notes/chords stored as {} then pickled at {}".format(type(notes), input_notes_file))
    print("First 20 notes/chords: {}".format(notes[:20]))
    return notes


def prepare_sequences(notes):
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


def reshape_for_training(network_input, network_output,n_patterns):
    print("\n**Reshaping for training**")
    # convert network input/output from lists to numpy arrays
    # reshape input to (notes less sequence length, sequence length)
    # reshape output to (notes less sequence length, unique notes/chords)
    network_input_r = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_output_r = np_utils.to_categorical(network_output)

    print("Reshaping network input to (notes - sequence length, sequence length) {}".format(network_input_r.shape))
    print("Reshaping network output to (notes - sequence length, unique notes) {}".format(network_output_r.shape))
    return network_input_r, network_output_r


def create_network(network_input, n_vocab, weights_file=weights_file):
    print("\n**LSTM model initializing**")
    # three layer model
    model = Sequential()
    timesteps = network_input.shape[1]
    data_dim = network_input.shape[2]
    model.add(LSTM(node1, input_shape=(timesteps, data_dim), return_sequences=True))
    model.add(Dropout(drop))
    model.add(LSTM(node1, return_sequences=True))
    model.add(Dropout(drop))
    model.add(LSTM(node1))
    model.add(Dense(node2))
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


def train_model(model, network_input_r, network_output_r, epochs, batch_size):
    # saves weights after each epoch
    weights_checkpoint = '../weights/weights-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(weights_checkpoint, monitor='loss',verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    print("Fitting Model. \nNetwork Input Shape: {} Network Output Shape: {}".format(network_input_r.shape,network_output_r.shape))
    print("Epochs: {} Batch Size: {}".format(epochs, batch_size))
    model.fit(network_input_r, network_output_r, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

    # saves weights upon training completion
    weights_file = '../weights/{}-{}-{}-{}-lstm_weights.hdf5'.format(timestamp, output_name, epochs, sequence_length)
    model.save_weights(weights_file)
    print("TRAINING complete - weights saved at: {}".format(weights_file))
    return model, weights_file


def reshape_for_creation(network_input, n_patterns, sequence_length, n_vocab):
    print("\n**Preparing sequences for output**")

    # the network input variables below are unshaped (pre-reshaped)
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length,1)) / float(n_vocab)
    print("Network Input of length {} is reshaped to normalized input of {}".format(len(network_input),normalized_input.shape))
    return normalized_input


def generate_notes(model, network_input, pitchnames,n_vocab):
    print("\n**Generating notes**")
    # convert integers back to notes/chords
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    print("Integer to note conversion at length: {}".format(len(int_to_note)))

    # randomly instantiate with single number from 0 to length of network input
    start = np.random.randint(0,len(network_input)-1)
    pattern = network_input[start]

    prediction_output = []
    print("Pattern begins with length {} and type {}".format(len(pattern),type(pattern)))
    # for each note in notes generated declared as hyperparameter above (ie 500)
    for note_index in range(notes_generated):
        prediction_input = np.reshape(pattern, (1,len(pattern),1)) / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print("Pattern ends with length {} and type {}".format(len(pattern),type(pattern)))
    print("Prediction Output Length: {}\nPrediction Output first 100: {}".format(len(prediction_output), prediction_output[:100]))
    return prediction_output


def create_midi(prediction_output):
    print("\n**Creating midi**")
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        # prepares chords (if) and notes (else)
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = sound
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = sound
            output_notes.append(new_note)
        offset += 0.5

    print("Generating {} notes stored as {}".format(len(output_notes),type(output_notes)))
    midi_stream = stream.Stream(output_notes)
    output_file = '../output/{}-{}-{}-{}-lstm_midi.mid'.format(timestamp, output_name, epochs, sequence_length)
    midi_stream.write('midi',fp=output_file)
    print("Midi saved at: {}".format(output_file))
    with open(output_notes_file, 'wb') as f:
        pickle.dump(output_notes, f)
    print("Output notes/chords stored as {} then pickled at {}".format(type(output_notes), output_notes_file))
    return output_notes

full_execution(midi_files)
