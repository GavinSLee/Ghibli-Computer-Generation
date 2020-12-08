import numpy as np
import glob
import pickle
from music21 import converter, instrument, note, chord
from keras import utils


def get_notes(directory_name):
    """  
    Parses through every MIDI file in the directory folder passed in and gets every single note, chord, and rest in the MIDI file. 

    :param directory_name: name of the directory which contains all of the midi files (str) 
    :returns: list of notes, chords, and rests that are in every MIDI file in the directory_name
    """

    notes = []

    for file in glob.glob(directory_name + "/*.mid"):

        curr_notes = []

        curr_midi = converter.parse(file)

        # Returns a list of instrument channels through which we need to parse through 
        channels = instrument.partitionByInstrument(curr_midi)

        # The current file has multiple instrument parts 
        if channels != None and len(channels.parts):
            for channel in channels.parts:
                curr_notes.append(channel.recurse())
        # The current file has only one instrument part, to which we parse only the flat notes 
        else:
            curr_notes.append(curr_midi.flat.notes) 

        for element in curr_notes:
            if isinstance(element, note.Rest):
                notes.append(str(element.name)  + " " + str(element.quarterLength))
            elif isinstance(element, note.Note):
                notes.append(str(element.pitch) + " " +  str(element.quarterLength))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder) + " " + str(element.quarterLength))
            
    with open(directory_name + '/saved_notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = utils.to_categorical(network_output)

    return (network_input, network_output)

if __name__ == "__main__":
    directory_name = "data"
    get_notes(directory_name)