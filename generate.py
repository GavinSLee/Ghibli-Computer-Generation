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
    Generates a list of indices given the model, inputs, vocab dictionary, start index, and the number of notes to generate. 

    :param model: RNN that is trained in model.py  
    :param inputs: list of lists of indices
    :param vocab: dictionary that maps each note to a unique index 
    :param start_index: the index for which we start with to generate notes (defaults to None, which will start with a random index) 
    :param num_generate: the number of notes to generate (defaults to 400 notes) 

    :return: list of predicted indices  
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
    Logic to parse rest. 
    """

    new_rest = note.Rest(pred_note)
    new_rest.offset = offset
    new_rest.storedInstrument = instrument.Piano() 
    return new_rest 

def parse_chord(pred_note, offset):
    """
    Helper method that parses a chord. 
    """

    notes_in_chord = pred_note.split('.')
    notes = []
    for current_note in notes_in_chord:
        new_note = note.Note(int(current_note))
        new_note.storedInstrument = instrument.Piano()
        notes.append(new_note)
    new_chord = chord.Chord(notes)
    new_chord.offset = offset
    return new_chord 

def parse_note(pred_note, offset):
    """
    Helper method that parses a regular note 
    """

    new_note = note.Note(pred_note)
    new_note.offset = offset
    new_note.storedInstrument = instrument.Piano()
    return new_note 


def generate_midi(predicted_notes):
    """ 
    Generates a midi file based the list of predicted notes passed in. 

    :param predicted_notes: list of predicted notes generated in generate_notes() 

    :return: None, the method is written such that it will write to generated_output.mid
    """
    offset = 0
    parsed_notes = []

    for pred_note in predicted_notes:
        pred_note = pred_note.split()
        temp = pred_note[0]
        duration = pred_note[1]
        pred_note = temp
        # Check rest
        if('rest' in pred_note):
            new_rest = parse_rest(pred_note, offset) 
            parsed_notes.append(new_rest)
        # Check for chords 
        elif ('.' in pred_note) or pred_note.isdigit():
            new_chord = parse_chord(pred_note, offset) 
            parsed_notes.append(new_chord)
        # Check for regular note 
        else:
            new_note = parse_note(pred_note, offset) 
            parsed_notes.append(new_note)

        offset += convert_to_float(duration)

    midi_stream = stream.Stream(parsed_notes)

    midi_stream.write('midi', fp='generated_output.mid')


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac



