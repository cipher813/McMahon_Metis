from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
import glob
import pickle
from music21 import converter, instrument, note, chord


from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io


midi_files = '../../data/Music/Tadpole/CelticMidis/ADEN.MID'

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
    assert note_count > 0
    n_vocab = len(set(notes))
    print("Loaded {} midi files {} notes and {} unique notes".format(note_count, len(notes), n_vocab))

    input_notes_file = 'input_notes'
    with open(input_notes_file, 'wb') as f:
        pickle.dump(notes, f)
    print("Input notes/chords stored as {} then pickled at {}".format(type(notes), input_notes_file))
    print("First 20 notes/chords: {}".format(notes[:20]))
    text = notes
    return text

notes = convert_midis_to_notes(midi_files)

unique_notes = sorted(list(set(notes)))
print('total chars:', len(unique_notes))
note_to_int = dict((n, i) for i, n in enumerate(unique_notes))
int_to_note = dict((i, n) for i, n in enumerate(unique_notes))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(notes) - maxlen, step):
    sentences.append(notes[i: i + maxlen])
    next_chars.append(notes[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = []
        sentence = text[start_index: start_index + maxlen]
        generated.append(sentence)
        # print('----- Generating with seed: "' + sentence + '"')
        # sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated.append(next_char)
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=1,
          callbacks=[print_callback])
