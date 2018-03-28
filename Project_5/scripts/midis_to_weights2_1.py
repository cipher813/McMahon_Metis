# Convert midi files to weights via training

import sys
import re
from datetime import datetime

from music21 import instrument

import processing2_1 as pr
import nn2_1 as nn
import generate2_1 as cr
import utils2_1 as ut

# midi_files = '../data/Music/Tadpole/CelticMidis/ADEN.MID'
midi_files = '../data/Music/Tadpole/**/*.MID'

timestamp = re.sub(r'[-: ]','',str(datetime.now()).split('.')[0])[:-2]
output_name = midi_files.split('/')[-3]
epochs = 200
batch_size = 1024
sequence_length = 100 # the LSTM RNN will consider this many notes
output_tag = '../output/{}-{}-{}-{}-'.format(timestamp, output_name, epochs, sequence_length)
sound = instrument.Bagpipes()
notes_generated = 500


def midis_to_weights(midi_files, output_tag, sequence_length):
    notes = pr.convert_midis_to_notes(midi_files, output_tag)
    network_input, network_output, n_patterns, n_vocab, pitchnames = pr.prepare_sequences(notes, sequence_length)
    network_input_r, network_output_r = pr.reshape_for_training(network_input, network_output,sequence_length)
    model = nn.create_network(network_input_r, n_vocab)
    model, weight_file = nn.train_model(model, network_input_r, network_output_r, epochs, batch_size, output_tag, sequence_length)
    return weight_file, notes


terminal_output = output_tag + 'terminal.log'
sys.stdout = ut.Logger(terminal_output)
print("Terminal output being saved at {}".format(terminal_output))
midis_to_weights(midi_files, output_tag, sequence_length)
print("Run Complete. Terminal log saved at {}".format(terminal_output))
