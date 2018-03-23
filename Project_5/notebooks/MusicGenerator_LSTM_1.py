
# coding: utf-8

# # Music Generator LSTM Neural Network Model

# ### Import Packages

# In[1]:


import sys
sys.executable


# In[2]:


# !pip install intervaltree


# In[3]:


import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from IPython.display import Audio
from intervaltree import Interval, IntervalTree

# from scipy.fftpack import fft


# ### Assumptions

# In[4]:


notes_file = '../data/notes'
midi_files = '../data/Bach/*.mid'
# midi_files = "../midi_songs/*.mid"
# MusicNet_file = '/media/cipher000/DATA/Dropbox/Programming/MusicNet/Dataset/musicnet.npz'

sequence_length = 100 # the lstm will predict the next note based on the last set of notes heard
node1 = 512
node2 = 256
drop = 0.3
epochs = 10 # 200
batch_size = 64
notes_generated = 500

fs = 44100      # samples/second
d = 2048        # input dimensions
m = 128         # number of notes
features = 0    # first element of (X,Y) data tuple
labels = 1      # second element of (X,Y) data tuple


# ### Load Midi Files from MusicNet

# ### Train Model

# In[5]:


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    notes_dict = {}
    cnt = 0

#     for file in train_data.files:
    for file in glob.glob(midi_files):
        print(file)
        notes_per_file = []
#         print("Type: {} Length: {} Contents: {}".format(type(file),len(file),file))
#         print("Y Type: {} Length: {} Contents: {}".format(type(Y),len(Y),Y))

        midi = converter.parse(file)
#         print(type(midi))
#         notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)
#         print(type(parts))
        if parts: # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                notes_per_file.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                to_append = '.'.join(str(n) for n in element.normalOrder)
                notes.append(to_append)
                notes_per_file.append(to_append)
        notes_dict[file] = notes_per_file
        cnt +=1
    with open(notes_file, 'wb') as filepath:
        pickle.dump(notes, filepath)
    print("{} midi files and {} notes".format(cnt,type(notes)))
    return notes,cnt


# In[6]:


notes, cnt = get_notes()


# In[ ]:


len(notes)


# In[ ]:


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """


    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
#         print("Sequence in: {}".format(sequence_in))
        sequence_out = notes[i + sequence_length]
#         print("Sequence out: {}".format(sequence_out))
        input_append = [note_to_int[char] for char in sequence_in]
        network_input.append(input_append)
#         print("Network input: {}".format(input_append))
        output_append = note_to_int[sequence_out]
        network_output.append(output_append)
#         print("Network output: {}".format(output_append))


    n_patterns = len(network_input)
    print("Number of Patterns: {}".format(n_patterns))

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    print("Network input reshaped: {}".format(network_input.shape))
    
    # normalize input
    network_input = network_input / float(n_vocab)
    print("Network input over float n_vocab: {}".format(network_input.shape))

    network_output = np_utils.to_categorical(network_output)
    print("Network output: {}".format(network_output.shape))

    return (network_input, network_output)


# In[ ]:


n_vocab = len(set(notes))

network_input, network_output = prepare_sequences(notes, n_vocab)


# In[ ]:


def create_network(network_input, n_vocab,weights_file=None):
    """ create the structure of the neural network """

    
    model = Sequential()
    model.add(LSTM(node1,input_shape=(network_input.shape[1], network_input.shape[2]),return_sequences=True))
    model.add(Dropout(drop))
    model.add(LSTM(node1, return_sequences=True))
    model.add(Dropout(drop))
    model.add(LSTM(node1))
    model.add(Dense(node2))
    model.add(Dropout(drop))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    # Load the weights to each node
    try:
        model.load_weights(weights_file)
        print("Weights file loaded: {}".format(weights_file))
    except Exception as e:
        print("Training mode (No weights file)")
#         print(e)
        pass

    return model


# In[ ]:


model = create_network(network_input, n_vocab)


# In[ ]:


def train(model, network_input, network_output):
    """ train the neural network """

    filepath = "../weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    weights_file = filepath
    print("Weights File Update to {}".format(weights_file))
#     print(filepath)
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)


# In[ ]:


train(model, network_input, network_output)


# ### Test and Create Midi File

# In[ ]:


weights_file = '../weights/weights.hdf5'


# In[ ]:


#load the notes used to train the model
with open(notes_file, 'rb') as filepath:
    notes = pickle.load(filepath)
print("Length of notes: {}".format(len(notes)))
# Get all pitch names
pitchnames = sorted(set(item for item in notes))
print("Length of Pitch Names: {}".format(len(pitchnames)))

# Get all pitch names
n_vocab = len(set(notes))
print("Length of N Vocab: {}".format(n_vocab))


# In[ ]:


def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
#     sequence_length = 100
    network_input = []
    output = []
    
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    print("Note to Integer Dictionary Length: {}".format(len(note_to_int)))

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])
#         print("Preparation {}: {}".format(i,note_to_int[sequence_out]))

    n_patterns = len(network_input)
    print("N Patterns Length: {}".format(n_patterns))

    # reshape the input into a format compatible with LSTM layers
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)
    print("Network Input Shape: {}  Normalized Input Shape: {}".format(len(network_input), len(normalized_input)))

    return (network_input, normalized_input)


# In[ ]:


network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)


# In[ ]:


model = create_network(normalized_input, n_vocab,weights_file)


# In[ ]:


def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_input)-1)
    print("Start: {}".format(start))

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    print("Int to Note Length: {}".format(len(int_to_note)))
    
    pattern = network_input[start]
    prediction_output = []
#     print("Pattern: {}  Prediction Output: {}".format(pattern, prediction_output))

    # generate 500 notes
    for note_index in range(notes_generated):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]
#         print("Pattern: {}".format(pattern))
#     print("Prediction Output: {}".format(prediction_output))
    print("Notes generated: {}".format(notes_generated))
    return prediction_output


# In[ ]:


prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)


# In[ ]:


def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
#     print("Prediction Output: {}".format(prediction_output))
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
#             print("Pattern: {} New Chord: {}".format(pattern, new_chord))
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
#             print("Pattern: {} New Note: {}".format(pattern, new_note))

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    print("Midi Stream: {}".format(midi_stream))
    output_file = '../output/{}{}.mid'.format(weights_file,epochs)

    midi_stream.write('midi', fp=output_file)


# In[ ]:


create_midi(prediction_output)


# ### Resources

# Model adapted from Towards Data Science blog [here](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)
# with accompanying "Classical Piano Composer" github repo [here](https://github.com/Skuldur/Classical-Piano-Composer).

# Download MusicNet from http://homes.cs.washington.edu/~thickstn/musicnet.html
