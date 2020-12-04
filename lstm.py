""" This module prepares midi file data and feeds it to the neural
    network for training """
import numpy as np
import matplotlib.pyplot as plt
import os

import glob
import pickle

from music21 import converter, instrument, stream, note, chord

#Run version 2.1.6
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Bidirectional, Flatten
from keras import utils
from keras.callbacks import ModelCheckpoint
from keras_self_attention import SeqSelfAttention
from tensorflow.python.keras.layers.core import Reshape
import tensorflow as tf

def train_network(notes, n_vocab):
    """ Train a Neural Network to generate music """
    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./full_set_beethoven_mozart directory. Call BEFORE train """
    notes = []

    for file in glob.glob("data/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi) #Change to only grab the piano???
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch) + " " +  str(element.quarterLength))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder) + " " + str(element.quarterLength))
            elif isinstance(element, note.Rest):
                notes.append(str(element.name)  + " " + str(element.quarterLength))

    with open('data/notes', 'wb') as filepath:
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

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.add(Bidirectional(LSTM(512,
        input_shape=(network_input.shape[1], network_input.shape[2]), #n_time_steps, n_features?
        return_sequences=True)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(0.3))
    
    model.add(LSTM(512,return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Flatten()) # Supposedly needed to fix stuff before dense layer
    
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = os.path.abspath("weights-{epoch:03d}-{loss:.4f}.hdf5")
    checkpoint = ModelCheckpoint(
        filepath,
        period=10, #Every 10 epochs
        monitor='loss',
        verbose=1,
        save_best_only=False,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=150, batch_size=128, callbacks=callbacks_list)

if __name__ == "__main__":    
    #load files in
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    #train
    train_network(notes, n_vocab)