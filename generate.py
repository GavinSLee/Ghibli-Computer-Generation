import numpy as np
from music21 import instrument, stream, note, chord

def convert_indices_to_notes(indices, vocab):
    """
    Converts a list of indices passed in back out to a list of their respective notes. 

    :param indices: list of indices (which ultimately represent notes) 
    :param vocab: dictionary that maps each note to a unique index 

    :returns: list of notes 
    """

    # Reverses the vocab so that each index is the key, and each value is the note 
    reverse_vocab = {value:key for key, value in vocab.items()}

    notes = [] 
    for i in range(len(indices)):
        note = reverse_vocab[indices[i]] 
        notes.append(note) 
    
    return notes 

def predict_next_index(music_model, vocab, sequence):
    """
    Predicts the next index (which will be used to predict the next note) given a sequence of indices. 

    :param model: the model trained in model.py 
    :param vocab: a dictionary that maps each note to a unique index 
    :param sequence: a sequence of 
    """

    prediction_input = np.reshape(sequence, (1, len(sequence), 1))
    prediction_input = prediction_input / float(len(vocab))

    probabilities = music_model.predict(prediction_input, verbose=0)
    predicted_index = np.argmax(probabilities)
 
    sequence = sequence[1:] + [predicted_index]

    return predicted_index, sequence 

def generate_notes(model, inputs, vocab, start_index = None, num_generate = 400):
    """
    Generates a list of notes given the model, inputs, vocab dictionary, start index, and the number of notes to generate. 

    :param model: RNN that is trained in model.py  
    :param inputs: list of lists of indices
    :param vocab: dictionary that maps each note to a unique index 
    :param start_index: the index for which we start with to generate notes (defaults to None, which  will start with a random index) 
    :param num_generate: the number of notes to generate (defaults to 400 notes) 

    :return: list of predicted notes 
    """

    music_model = model.make_model_1()

    if start_index == None:
        start_index = np.random.randint(0, len(inputs) - 1)

    sequence = inputs[start_index]
    pred_indices = []

    for i in range(num_generate):
        curr_pred_index, sequence = predict_next_index(music_model, vocab, sequence)
        pred_indices.append(curr_pred_index)  

    pred_notes = convert_indices_to_notes(pred_indices, vocab) 
    return pred_notes 

def parse_rest(pred_note, offset):
    """
    Handles the logic for parsing a rest. Here, we set the offset and the instrument of the rest. Used in conjunction with generating a MIDI file. 

    :param pred_note: the predicted note that is checked as a rest. 
    :param: offset to ensure notes don't stack on top of one another. 

    :return: parsed rest note 
    """

    parsed_rest = note.Rest(pred_note)
    parsed_rest.offset = offset
    parsed_rest.storedInstrument = instrument.Piano() 
    return parsed_rest 

def parse_chord(pred_note, offset):
    """
    Handles the logic for parsing a chord. Here, we set the offset of the chord as well. Used in conjunction with generating a MIDI file. 

    :param pred_note: the predicted note that is checked as a chord. 
    :param: offset to ensure notes don't stack on top of one another. 

    :return: parsed chord  
    """

    notes_in_chord = pred_note.split('.')
    notes = []

    # Note that chords have multiple parts to it when parsed using Music21. We must parse through every single one of these notes in the chord and store it in a list. 

    for curr_note in notes_in_chord:
        new_note = note.Note(int(curr_note))
        new_note.storedInstrument = instrument.Piano()
        notes.append(new_note)
    parsed_chord = chord.Chord(notes)
    parsed_chord.offset = offset
    return parsed_chord 

def parse_note(pred_note, offset):
    """
    Handles the logic for parsing a regular note. Here, we set the offset and the kind of instrument used. Used in conjunction with generating a MIDI file. 

    :param pred_note: the predicted note that is checked as a flat note. 
    :param: offset to ensure notes don't stack on top of one another. 

    :return: parsed flat note 
    """

    parsed_note = note.Note(pred_note)
    parsed_note.offset = offset
    parsed_note.storedInstrument = instrument.Piano()
    return parsed_note 


def generate_midi(predicted_notes):
    """ 
    Generates a midi file based on the list of predicted notes passed in. 

    :param predicted_notes: list of predicted notes generated in generate_notes() 

    :return: None, the method is written such that it will write to generated_output.mid
    """
    offset = 0
    parsed_notes = []

    for curr_note in predicted_notes:
        note_objects = curr_note.split()
        pred_note = note_objects[0]
        duration = note_objects[1]

        # Check rest
        if('rest' in pred_note):
            parsed_rest = parse_rest(pred_note, offset) 
            parsed_notes.append(parsed_rest)
        # Check for chords 
        elif ('.' in pred_note) or pred_note.isdigit():
            parsed_chord = parse_chord(pred_note, offset) 
            parsed_notes.append(parsed_chord)
        # Check for regular note 
        else:
            new_note = parse_note(pred_note, offset) 
            parsed_notes.append(new_note)

        offset += float(duration) 

    midi_stream = stream.Stream(parsed_notes)

    midi_stream.write('midi', fp='generated_output.mid')


# def convert_to_float(frac_str):
#     try:
#         return float(frac_str)
#     except ValueError:
#         num, denom = frac_str.split('/')
#         try:
#             leading, num = num.split(' ')
#             whole = float(leading)
#         except ValueError:
#             whole = 0
#         frac = float(num) / float(denom)
#         return whole - frac if whole < 0 else whole + frac



