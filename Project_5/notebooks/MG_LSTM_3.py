
# coding: utf-8

# # MG LSTM

# In[1]:


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
# from tensorflow.python.client import device_lib

print(sys.executable)
print(K.tensorflow_backend._get_available_gpus())


# In[2]:


notes_file = '../data/notes'
midi_files = '/home/ubuntu/Music/MidiWorld/Dance/*.mid'
weights_file = '../weights/lstm_weights.hdf5'

output_name = midi_files.split('/')[-2]

timestamp = str(datetime.datetime.now()).split()[0].replace('-','')


# In[3]:


sequence_length = 100 # the lstm will predict the next note based on the last set of notes heard
node1 = 512
node2 = 256
drop = 0.3
epochs = 50 # 200
batch_size = 64
notes_generated = 500


# ### Training

# In[4]:


def convert_to_notes():
    notes = []
    notes_dict = {}
    cnt = 0
    
    print("Loading Midi files")
    for file in glob.glob(midi_files):
        print(file)
        notes_per_file = []
        try:
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)

            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
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
        except Exception as e:
            print(e)
            pass
    with open(notes_file, 'wb') as filepath:
        pickle.dump(notes, filepath)
    print("{} midi files and {} notes".format(cnt,len(notes)))
    print("Notes Converted")
    n_vocab = len(set(notes))
    return notes, n_vocab, cnt

def prep_train_sequences(notes, n_vocab):
    print("Preparing sequences for training")
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    network_input = []
    network_output = []
    
    for i in range(0,len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        
        input_append = [note_to_int[char] for char in sequence_in]
        network_input.append(input_append)
        output_append = note_to_int[sequence_out]
        network_output.append(output_append)
        
    n_patterns = len(network_input)
    
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    network_output = np_utils.to_categorical(network_output)
    print("Sequences Prepared")
    return pitchnames, network_input, network_output

def create_network(network_input, n_vocab,weights_file=None):
    print("Keras model initializing")
    model = Sequential()
    # Layer 1
    model.add(LSTM(node1,input_shape=(network_input.shape[1], network_input.shape[2]),return_sequences=True))
    model.add(Dropout(drop))
    # Layer 2
    model.add(LSTM(node1, return_sequences=True))
    model.add(Dropout(drop))
    # Layer 3
    model.add(LSTM(node1))
    model.add(Dense(node2))
    model.add(Dropout(drop))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if weights_file:
        model.load_weights(weights_file)
        print("LSTM model initialized for midi creation with weights from {}".format(weights_file))
    else:
        print("LSTM model initialized for training - no weights file")
    
    return model

def train(model, network_input, network_output):
    print("Training LSTM network")
    filepath = "../weights/weights-{epoch:02d}-{loss:.4f}.hdf5"
    
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min')
    
    callbacks_list = [checkpoint]
    
    model.fit(
        network_input, 
        network_output, 
        epochs=epochs, 
        batch_size=batch_size, 
        callbacks=callbacks_list)
    weights_file = '../weights/{}-{}-lstm_weights.hdf5'.format(timestamp, output_name)
    model.save_weights(weights_file)
    print("LSTM training complete - weights saved at: {}".format(weights_file))
    return weights_file


# ### Create MIDI

# In[5]:


def prep_output_sequences(notes, pitchnames, n_vocab):
    print("Preparing sequences for output")
    network_input = []
    network_output = []
    
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
        
    n_patterns = len(network_input)
    
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    normalized_input = normalized_input / float(n_vocab)
    
    return network_input, normalized_input

def generate_notes(model, network_input, pitchnames, n_vocab):
    print("Generating notes")
    start = np.random.randint(0,len(network_input)-1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    
    pattern = network_input[start]
    prediction_output = []
    
    for note_index in range(notes_generated):
        prediction_input = np.reshape(pattern, (1,len(pattern),1))
        prediction_input = prediction_input / float(n_vocab)
        
        prediction = model.predict(prediction_input, verbose=0)
        
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        
    return prediction_output

def create_midi(prediction_output,output_name, epochs):
    print("Creating midi")
    offset = 0
    output_notes = []
    
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Flute()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        
        offset += 0.5
        
    midi_stream = stream.Stream(output_notes)
    output_file = '../output/{}-{}-lstm_midi-{}.mid'.format(timestamp,output_name,epochs)
    midi_stream.write('midi',fp=output_file)
    print("Midi saved at: {}".format(output_file))
    return midi_stream


# ### LSTM Model Execution

# In[6]:


def train_create_midi():
    # Midi preparation
    notes, n_vocab, cnt = convert_to_notes()
    pitchnames, network_input, network_output = prep_train_sequences(notes, n_vocab)
    # LSTM training
    model = create_network(network_input, n_vocab)
    weights_file = train(model, network_input, network_output)
    # Midi creation
    network_input, normalized_input = prep_output_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab,weights_file)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    midi = create_midi(prediction_output,output_name,epochs)


# In[7]:


train_create_midi()


# ### Resources

# Model adapted from Sigurour Skuli's [How to Generate Music using a LSTM Neural Network in Keras](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)
