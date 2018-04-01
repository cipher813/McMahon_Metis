import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
from datetime import datetime
import re
import pickle
# from music21 import instrument

import processing as pr
import neural_network as nn
import generate as cr
import utils as ut

weight_file = 'output/201803311943-Tadpole-1-200-weight-01-3.9807-8.1616.hdf5'
note_file = 'output/201803311943-Tadpole-1-200-input_notes'

timestamp = re.sub(r'[-: ]','',str(datetime.now()).split('.')[0])[:-2]
# output_name = midi_files.split('/')[-3]
# epochs = 200
# batch_size = 128
sequence_length = 200 # the LSTM RNN will consider this many notes
output_tag = 'output/{}-{}-'.format(timestamp, sequence_length)
notes_generated = 500

# convert fully trained weights to midi file
def weights_to_midi(note_file, sequence_length, weight_file):
    with open(note_file, 'rb') as filepath:
        notes = pickle.load(filepath)
    network_input, network_output, n_patterns, n_vocab, pitchnames = pr.prepare_sequences(notes, sequence_length)
    normalized_input = pr.reshape_for_creation(network_input, n_patterns, sequence_length, n_vocab)
    model = nn.create_network(normalized_input, n_vocab, weight_file)
    prediction_output= cr.generate_notes(model, network_input, pitchnames,notes_generated, n_vocab)
    output_notes = cr.create_midi(prediction_output, output_tag, sequence_length)
    return output_notes


terminal_output = output_tag + 'terminal.log'
sys.stdout = ut.Logger(terminal_output)
print("Terminal output being saved at {}".format(terminal_output))
weights_to_midi(note_file, sequence_length, weight_file)
print("Run Complete. Terminal log saved at {}".format(terminal_output))
