from music21 import converter, instrument, note, chord
from tensorflow.keras import utils
import numpy as np
import glob
import pickle

def midi_to_notes(fileDirectory):
    """
    Gets all the notes and chords of the Midi files in the file directory passed in and returns a list of those notes and chords. 

    :param fileDirectory: the name of the file directory that contains the midi files (str) 
    :return: an array of notes, [str]
    """

    midi_filelink = fileDirectory + "/*.mid" #adding midi ending 
    notes = []
    
    # Loop through all the files in the file directory  
    for file in glob.glob(midi_filelink):
        midi_file = converter.parse(file)
        parts = instrument.partitionByInstrument(midi_file)

        # Interestingly, this condition occurs and we get Fire Emblem (which is a mislabeling). Note that all the MIDI files are single instrument, so regardless, our MIDI files are still uniform. 
        if parts != None:
            notes_to_parse = parts.parts[0].recurse() 
        else:
            notes_to_parse = midi_file.flat.notes
        # Parse through all the notes in the current MIDI file. Notes can be either notes or chords. 
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    # Pickle the data at the end, so we can get notes when testing
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes 

def get_notes_sequences(notes, vocab_size): 
    """
    Create inputs and labels. 
    
    :param notes: array that contains all of the notes contained in the midi files. 
    :param vocab_size: number of unique notes and chords in notes. 
    :return: A tuple of inputs and labels 
    """
    sequence_length = 100
    
    # Create a dictionary to map unique notes to an index 
    note_dict = {}
    index = 0  
    for note in notes:
        if note not in note_dict:
            note_dict[note] = index 
            index += 1

    inputs = []
    labels = []

    # Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        inputs.append([note_to_int[char] for char in sequence_in])
        labels.append(note_to_int[sequence_out])

    n_patterns = len(inputs)

    # Reshape the input into a format compatible with LSTM layers
    inputs = np.reshape(inputs, (n_patterns, sequence_length, 1))
    # Normalize input
    inputs = inputs / float(vocab_size)
    labels = utils.to_categorical(labels)

    return (note_to_int, inputs, labels)

def predict_notes(model, note_to_int, starting_note = None, num_notes_generate = 1000):
    """
    Converts the notes array passed in to a MIDI file, which can be played using a program such as Synthesia. 

    :param note_to_int: dictionary that maps note to its respective index
    :param starting_note: starting note to start the computer generated song; if none, then a starting note is randomly selected 
    param num_notes_generate: generates the number of notes passed in 

    :return: midi file 
    """
    
    if starting_note == None:
        starting_note = np.random.randint(0, len(note_to_int))
    

    # List that will hold our predictions 
    predictions = [] 


    for i in range(num_notes_generate):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        
def generate_midi(generated_notes):
    
    pass 