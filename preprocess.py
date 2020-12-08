import numpy as np
import glob
import pickle
from music21 import converter, instrument, note, chord
from keras import utils


def get_notes(directory_name):
    """  
    Parses through every MIDI file in the directory folder passed in and gets every single note, chord, and rest in each MIDI file. Returns a list of these notes, chords, and rests. 

    :param directory_name: name of the directory which contains all of the midi files (str) 
    :returns: list of notes, chords, and rests that are in every MIDI file in directory_name
    """

    notes = []

    file_list = glob.glob(directory_name + "/*.mid")

    for file in file_list:
        
        # Parse through the current file using the converter
        curr_midi = converter.parse(file)

        # Returns a list of instrument channels through which we need to parse through 
        channels = instrument.partitionByInstrument(curr_midi)

        # Get a list of all the note iterators when parsing through the channels
        notes_iterators = []

        # The current file has multiple instrument parts 
        if channels != None and len(channels.parts):
            for part in channels.parts:
                notes_iterators.append(part.recurse())
        # The current file has only one instrument part, to which we parse only the flat notes 
        else:   
            notes_iterators.append(curr_midi.flat.notes)

        # Loop through all the iterators, for each iterator, check if the note is either a rest, note, or chord.
        for iterator in notes_iterators:
            for element in iterator:
                if isinstance(element, note.Rest):
                    notes.append(str(element.name)  + " " + str(element.quarterLength))
                elif isinstance(element, note.Note):
                    notes.append(str(element.pitch) + " " +  str(element.quarterLength))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder) + " " + str(element.quarterLength))
            break 
                
    with open(directory_name + '/saved_notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def get_vocab(notes):
    """
    Creates the dictionary that maps each note to a unique index. 

    :param notes: list of notes, chords, and rests that is returned by the get_notes method
    :return: a dictionary that maps each note in notes to a unique index 
    """

    vocab = {} 
    index = 0 

    # Loop through each note in notes, checking if the note exists in the dictionary. If it doesn't exist in the dictionary, map the note to a unique index and increment the index by one. 
    for note in notes:
        if note not in vocab:
            vocab[note] = index 
            index += 1

    return vocab  

def get_windows(notes, window_size = 50):
    """ 
    Creates the input windows (sequences) of the notes passed in and the following note for each window based on the window_size parameter. 

    :param notes: list of notes, chords, and rests that is returned by the get_notes method
    :param window_size: the length of the sequences  
    
    :return: a tuple of (input_windows, label_windows), where each window in input_windows is a window of input notes, and label_windows is the corresponding sequential note for each window. 
    """

    input_windows = [] 
    label_notes = [] 
    
    # loop through each note in notes and create windows of notes of window_size, with its corresponding next note, which represents the "label". 
    for i in range(0, len(notes) - window_size, 1):
        curr_input_window = notes[i:i + window_size]
        input_windows.append(curr_input_window) 

        curr_label_note = notes[i + window_size]
        label_notes.append(curr_label_note) 

    return input_windows, label_notes


def get_inputs_and_labels(input_windows, label_notes, vocab):
    """ 
    Uses the vocab dictionary passed in to "indicize" each of the notes in each window in input_windows and each note in label_notes. We return the same windows and labels, except each note is now its corresponding unique index.

    :param input_windows: list of lists, where each sublist is a window of notes of some window_size
    :param label_windows: list of notes, where each note is the sequential note for its corresponding window 
    :param vocab: dictionary that maps each note in notes to a unique index 
    
    :return: inputs, labels, a tuple where inputs is a list of lists of indices and labels is a list of indices, in which each index is the following index (note) in inputs.  
    """

    inputs = []
    labels = []

    # Loop through each window and use the vocab dictionary to map each word to its unique index. 
    for i in range(len(input_windows)):
        label_index = vocab[label_notes[i]]
        labels.append(label_index) 

        curr_input_window = input_windows[i] 
        input_indices = [] 

        for note in curr_input_window:
            input_index = vocab[note] 
            input_indices.append(input_index) 
        
        inputs.append(input_indices) 
    
    return inputs, labels 


def shape_inputs_and_labels(inputs, labels, vocab):
    """ 
    Shapes the inputs and labels so that they work with our model. Also normalizes our inputs such that each index is a value between 0 and 1, and we categorize our labels. 
    
    :param inputs: list of lists of indices that represent our input sequences
    :param labels: list of indices, in which each index is the following index (note) in inputs.

    :return: reshaped inputs and labels  
    """

    inputs = (np.reshape(inputs, (len(inputs), len(inputs[0]), 1))) / float(len(vocab)) 

    labels = utils.to_categorical(labels)

    return inputs, labels 

def final_inputs_and_labels(notes, vocab):
    """ 
    Takes all of the methods defined above and returns the final inputs and labels 

    :return: final inputs and labels to be passed into the model   
    """

    input_windows, input_labels = get_windows(notes)
    inputs, labels = get_inputs_and_labels(input_windows, input_labels, vocab)
    inputs, labels = shape_inputs_and_labels(inputs, labels, vocab) 

    return inputs, labels 
    