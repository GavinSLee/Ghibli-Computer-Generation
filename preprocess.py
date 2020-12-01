from music21 import converter, instrument, note, chord
import tensorflow as tf
from tensorflow.keras import utils
import numpy as np
import glob
import pickle

def midi_to_notes(fileDirectory: str):
    midi_filelink = fileDirectory + "/*.mid" #adding midi ending 
    notes = []
    for file in glob.glob(midi_filelink):
        midi_file = converter.parse(file)
        parts = instrument.partitionByInstrument(midi_file)
        raw_notes = None
        notes_to_parse = midi_file.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    #pickle the data at the end, so we can get notes when testing
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes 

def notes_to_midi():
    pass

def get_notes_sequences(notes, n_vocab): 
    """create inputs and labels"""
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    inputs = []
    labels = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        inputs.append([note_to_int[char] for char in sequence_in])
        labels.append(note_to_int[sequence_out])

    n_patterns = len(inputs)

    # reshape the input into a format compatible with LSTM layers
    inputs = np.reshape(inputs, (n_patterns, sequence_length, 1))
    # normalize input
    inputs = inputs / float(n_vocab)
    labels = utils.to_categorical(labels)

    return (inputs, labels)



