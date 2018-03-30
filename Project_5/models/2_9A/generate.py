import pickle
import numpy as np

from music21 import note, chord, stream, instrument


# sample function from Keras Nietsche example
# def sample(preds, temperature=1.0):
#     # helper function to sample an index from a probability array
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     # probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(preds)


def generate_notes(model, network_input, pitchnames,notes_generated, n_vocab):
    # diversity_list = [0.2,0.5,1.0,1.2]
    print("\n**Generating notes**")
    # convert integers back to notes/chords
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    # note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    print("Integer to note conversion at length: {}".format(len(int_to_note)))

    # randomly instantiate with single number from 0 to length of network input
    # network_input = network_input[1:]
    start = np.random.randint(0,len(network_input)-1)
    # for diversity in [0.2, 0.5, 1.0,1.2]:

    # generated = ''
    # pattern = network_input[start: start + 100]
    # generated += str(pattern)

    pattern = network_input[start]
    # #
    prediction_output = []
    # print("Pattern begins with length {} and type {}".format(len(pattern),type(pattern)))
    # print("Pattern: {}".format(pattern))
    # # for each note in notes generated declared as hyperparameter above (ie 500)
    for note_index in range(notes_generated):
        # x_pred = np.zeros((1,100,n_vocab))
        # for t, char in enumerate(pattern):
        #     print("T: {} Char: {}".format(t,char))
        #     x_pred[0,t,note_to_int[char]] = 1.
        # pattern = sample(pattern, diversity)
        prediction_input = np.reshape(pattern, (1,len(pattern),1)) / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        # diversity = diversity_list[np.random.randint(0,4)]
        # index = sample(prediction,diversity)
        index = np.argmax(prediction)
        result = int_to_note[index]

        # prediction_output += result
        prediction_output.append(result)
        #
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        # print("Types. Note index: {} prediction_input: {} prediction: {} index: {} result: {}".format(type(note_index),type(prediction_input),type(prediction),type(index),type(result)))

    print("Pattern ends with length {} and type {}".format(len(pattern),type(pattern)))
    print("Generated Note Length: {}\nFirst 100: {}".format(len(prediction_output), prediction_output[:100]))
    return prediction_output


def create_midi(prediction_output, output_tag, sequence_length):
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
                new_note.storedInstrument = instrument.Piano()
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

    print("Generating {} notes stored as {}".format(len(output_notes),type(output_notes)))
    midi_stream = stream.Stream(output_notes)
    midi_file = output_tag + 'lstm_midi.mid'
    midi_stream.write('midi',fp=midi_file)
    print("Midi saved at: {}".format(midi_file))

    output_notes_file = output_tag + 'output_notes'
    with open(output_notes_file, 'wb') as f:
        pickle.dump(output_notes, f)
    print("Output notes/chords stored as {} then pickled at {}".format(type(output_notes), output_notes_file))
    return output_notes, midi_file
