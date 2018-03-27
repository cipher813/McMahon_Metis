# Provides functionality for:
# 1. Train from midi files to (a) fully trained model and (b) midi creation
# 2. Train from partially trained model to (a) fully trained model and (b) midi creation
# 3. From fully trained model to midi creation
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pickle
import sys
import re
from datetime import datetime

from music21 import instrument

import processing2_0 as pr
import nn2_0 as nn
import create2_0 as cr
import utils2_0 as ut

# midi_files = '../data/Music/Tadpole/CelticMidis/ADEN.MID'
# midi_files = '../data/Music/Tadpole/**/*.MID'

timestamp = re.sub(r'[-: ]','',str(datetime.now()).split('.')[0])[:-2]
# output_name = midi_files.split('/')[-3]
# epochs = 200
# batch_size = 128
sequence_length = 100 # the LSTM RNN will consider this many notes
output_tag = '../output/{}-{}-'.format(timestamp, sequence_length)
sound = instrument.Bagpipes()
notes_generated = 500

model_file = '../output/archive/201803262134-model-71-0.0920-5.8716.hdf5'
notes_file = '../output/archive/201803262134-Tadpole-200-100-input_notes'

# convert midi files to (a) fully trained model, and (b) to midi file
# def midis_to_midi(midi_files, output_tag):
#     model_file, n_vocab, notes = midis_to_model(midi_files, output_tag)
#     output_notes = model_to_midi(notes, sequence_length, n_vocab,model_file)
#     return output_notes
#
# # convert midi files to fully trained model
# def midis_to_model(midi_files, output_tag):
#     notes = pr.convert_midis_to_notes(midi_files, output_tag)
#     network_input, network_output, n_patterns, n_vocab, pitchnames = pr.prepare_sequences(notes, sequence_length)
#     network_input_r, network_output_r = pr.reshape_for_training(network_input, network_output,sequence_length)
#     model = nn.create_network(network_input_r, n_vocab)
#     model_file = nn.train_model(model, network_input_r, network_output_r, epochs, batch_size, output_tag, sequence_length)
#     return model_file, n_vocab, notes


# convert fully trained model to midi file
def model_to_midi(notes_file, sequence_length, model_file, output_tag):
    model = nn.load_saved_model(model_file)
    with open(notes_file, 'rb') as filepath:
        notes = pickle.load(filepath)
    assert len(notes) > 0
    network_input, network_output, n_patterns, n_vocab, pitchnames = pr.prepare_sequences(notes, sequence_length)
    normalized_input = pr.reshape_for_creation(network_input, sequence_length, n_vocab)
    # model = nn.create_network(normalized_input, n_vocab, model_file)
    prediction_output= cr.generate_notes(model, normalized_input, pitchnames,notes_generated, n_vocab)
    output_notes = cr.create_midi(prediction_output, output_tag, sequence_length, sound)
    return output_notes


terminal_output = output_tag + 'terminal.log'
sys.stdout = ut.Logger(terminal_output)
print("Terminal output being saved at {}".format(terminal_output))
model_to_midi(notes_file, sequence_length, model_file, output_tag)
print("Run Complete. Terminal log saved at {}".format(terminal_output))
