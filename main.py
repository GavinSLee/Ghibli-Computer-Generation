from preprocess import midi_to_notes, predict_notes, generate_midi, get_notes_sequences
# from music_model import Model, train, test
import tensorflow as tf;
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord, stream 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.python.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint

def create_network(network_input, n_vocab, load_weights = False):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    if load_weights:
        model.load_weights('saved_weights.hdf5')
    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=1, batch_size=128, callbacks=callbacks_list)

def main():
    print("Starting preprocessing")
    directory_name = "data"
    notes = midi_to_notes(directory_name) 
    network_inputs_1, network_outputs, network_inputs_2, normalized_inputs, vocab_notes = get_notes_sequences(notes)
    print("Preprocessing complete")

    #custom model
    # model = Model(vocab_size)
    # for i in range(200):
    #     print(i)
    #     train(model, train_inputs, train_outputs)

    # sequential model
    vocab_size = len(vocab_notes) 
    model = create_network(normalized_inputs, vocab_size, load_weights = True)

    # train(model, network_inputs_1, network_outputs)
    predicted_notes = predict_notes(model, network_inputs_2, vocab_notes, starting_note = None, num_notes_generate = 150)
    generate_midi(predicted_notes) 
    print("Finished")
    #for test, maybe 80 - 20 split?
    

if __name__ == "__main__":
    main() 
