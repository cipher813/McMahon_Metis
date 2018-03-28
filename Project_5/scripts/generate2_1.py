import pickle
import numpy as np

from music21 import note, chord, stream

def generate_notes(model, network_input, pitchnames,notes_generated, n_vocab):
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
    print("Generated Note Length: {}\nFirst 100: {}".format(len(prediction_output), prediction_output[:100]))
    return prediction_output


def create_midi(prediction_output, output_tag, sequence_length, sound):
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
    output_file = output_tag + 'lstm_midi.mid'
    midi_stream.write('midi',fp=output_file)
    print("Midi saved at: {}".format(output_file))

    output_notes_file = output_tag + 'output_notes'
    with open(output_notes_file, 'wb') as f:
        pickle.dump(output_notes, f)
    print("Output notes/chords stored as {} then pickled at {}".format(type(output_notes), output_notes_file))
    return output_notes
