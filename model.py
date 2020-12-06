import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Bidirectional, Flatten
from keras.callbacks import ModelCheckpoint
from keras_self_attention import SeqSelfAttention



class Model:
    def __init__(self, network_input, vocab_size):
        """ create the structure of the neural network """
        self.learning_rate = 0.001
        self.dropout_rate = 0.3
        self.hidden_size = 512
        self.vocab_size = vocab_size
        self.input_shape = (network_input.shape[1], network_input.shape[2])

        self.epoch_size = 150
        self.batch_size = 128
    
    def make_model(self):
        model = Sequential()

        opt = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        model.add(Bidirectional(LSTM(self.hidden_size,
            input_shape=(self.input_shape), #n_time_steps, n_features?
            return_sequences=True)))

        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(Dropout(self.dropout_rate))
        
        model.add(LSTM(self.hidden_size,return_sequences=False))
        model.add(Dropout(self.dropout_rate))

        model.add(Flatten()) # Supposedly needed to fix stuff before dense layer
        
        model.add(Dense(self.vocab_size))

        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=opt)

        return model

    

    
def train(model, network_input, network_output):
    sequential_model = model.make_model()

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
    sequential_model.fit(network_input, network_output, epochs=model.epoch_size, batch_size=model.batch_size, callbacks=callbacks_list, shuffle=True)





