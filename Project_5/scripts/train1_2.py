# if to force run on CPU; else comment out
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from music21 import converter, instrument, note, chord, stream
from datetime import datetime
import sys
from keras import backend as K
import weight_generator1_2 as wg
import midi_generator1_2 as mg
import re
import pickle

print(sys.executable)
print("Available GPUs: {}".format(K.tensorflow_backend._get_available_gpus()))

# Hyperparameters
sequence_length = 100 # the LSTM RNN will consider this many notes
epochs = 1
batch_size = 128 # from 64
notes_generated = 500 # output will contain this many notes
midi_quantity = 5000 # number of midi files to load from dataset
sound = instrument.Piano() # declare a music21 package instrument

start_time = datetime.now()
timestamp = re.sub(r'[-: ]','',str(start_time).split('.')[0])[:-2]

# midi_files = '/media/cipher000/DATA/Music/Tadpole/CelticMidis/30BOTHUG.MID'
midi_files = '../data/Music/Tadpole/**/*.MID'
output_name = midi_files.split('/')[-3]
input_notes_file = '../output/{}-{}-{}-{}-input_notes'.format(timestamp, output_name, epochs, sequence_length)
output_notes_file = '../output/{}-{}-{}-{}-output_notes'.format(timestamp, output_name, epochs, sequence_length)
terminal_output = '../output/{}-{}-{}-{}-terminal.log'.format(timestamp, output_name, epochs, sequence_length)

# input_notes_file = '/media/cipher000/DATA/Dropbox/Programming/GitClones/Classical-Piano-Composer/data/notes'
# specify None for training, or weights file to move straight to midi creation.
# weights_file = '/media/cipher000/DATA/Dropbox/Programming/GitClones/Classical-Piano-Composer/weights/weights.hdf5'
# weights_file = '../weights/201803250911-Pop-1-100-lstm_weights.hdf5'
weights_file = None


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
def full_execution(weights_file, input_notes_file, midi_files, sequence_length, epochs, batch_size,start_time, midi_quantity, timestamp, output_name,notes_generated):
    if weights_file == None:
        # weights file generated and passed to midi generator
        notes, weights_file = wg.weight_generator(midi_files,
                                                    sequence_length,
                                                    epochs,
                                                    batch_size,
                                                    input_notes_file,
                                                    weights_file,
                                                    midi_quantity,
                                                    timestamp,
                                                    output_name)
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
    else:
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
full_execution(weights_file, input_notes_file, midi_files, sequence_length, epochs, batch_size,start_time, midi_quantity, timestamp, output_name,notes_generated)
print("Run Complete. Terminal log saved at {}".format(terminal_output))
