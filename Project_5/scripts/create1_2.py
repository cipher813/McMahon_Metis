from music21 import converter, instrument, note, chord, stream
from datetime import datetime
import weight_generator1_1 as wg
import midi_generator1_1 as mg
import pickle

import sys
import os
import re

# if to force run on CPU; else comment out
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras import backend as K

print(sys.executable)
print("Available GPUs: {}".format(K.tensorflow_backend._get_available_gpus()))

# Hyperparameters
sequence_length = 100 # the LSTM RNN will consider this many notes
epochs = 200
batch_size = 128 # from 64
notes_generated = 500 # output will contain this many notes
sound = instrument.Piano() # declare a music21 package instrument

start_time = datetime.now()
timestamp = re.sub(r'[-: ]','',str(start_time).split('.')[0])[:-2]

input_notes_file = '../output/archive/201803252146-Tadpole-200-100-input_notes'
weights_file = '../output/archive/201803252146-weights-25-0.2244.hdf5'

# midi_files = '../data/Music/Tadpole/**/*.MID'
output_name = input_notes_file.split('-')[-4]
# input_notes_file = '../output/{}-{}-{}-{}-input_notes'.format(timestamp, output_name, epochs, sequence_length)
output_notes_file = '../output/{}-{}-{}-{}-output_notes'.format(timestamp, output_name, epochs, sequence_length)
terminal_output = '../output/{}-{}-{}-{}-terminal.log'.format(timestamp, output_name, epochs, sequence_length)

# input weights and notes, output weights
def full_execution(weights_file, input_notes_file, sequence_length, epochs, batch_size,start_time, timestamp, output_name,notes_generated):
    with open(input_notes_file, 'rb') as filepath:
        notes = pickle.load(filepath)
    network_input, network_output, n_patterns, n_vocab, pitchnames = wg.prepare_sequences(notes, sequence_length)
    output_notes = mg.midi_generator(notes,
                                        sequence_length,
                                        epochs,
                                        batch_size,
                                        timestamp,
                                        output_name,
                                        weights_file,
                                        output_notes_file,
                                        notes_generated,
                                        sound)

    print("Script run time: {}".format(datetime.now() - start_time))

# Keras also has CSVLogger
class Logger(object):

    def __init__(self, terminal_output):
        self.terminal = sys.stdout
        self.log = open(terminal_output, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

sys.stdout = Logger(terminal_output)
print("Terminal output being saved at {}".format(terminal_output))
full_execution(weights_file, input_notes_file, sequence_length, epochs, batch_size,start_time, timestamp, output_name,notes_generated)
print("Run Complete. Terminal log saved at {}".format(terminal_output))
