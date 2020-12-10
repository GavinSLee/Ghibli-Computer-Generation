import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Bidirectional, Flatten
from keras.callbacks import ModelCheckpoint
from keras_self_attention import SeqSelfAttention

class Model:
    def __init__(self, inputs, vocab_size,  weights_path = None):
        """ 
        Creates needed values for the neural network; here, we set the hyperparameters of our model, such as the learning rate, batch size, etc.

        :param inputs: the list of lists of sequences that we pass to the network 
        :param vocab_size: the number of unique notes that are seen throughout all the MIDI files  
        :param weights: the name of the hdf5 file that contains the weights (str) 

        :return: None 
        """

        self.learning_rate = 0.01
        self.dropout_rate = 0.3
        self.hidden_size = 512
        self.vocab_size = vocab_size
        self.input_shape = (inputs.shape[1], inputs.shape[2])
        self.epoch_size = 120
        self.batch_size = 128
        self.weights_path = weights_path
    
    def make_model_1(self):
        """ 
        Makes the LSTM + Attention model. This model is: LSTM -> Attention -> LSTM  -> Dense -> Relu -> Dense -> Softmax.

        :return: None 
        """
        model = Sequential()

        model.add(Bidirectional(LSTM(self.hidden_size,return_sequences=True),input_shape=(self.input_shape[0], self.input_shape[1]))) 
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(Dropout(self.dropout_rate))
        
        model.add(LSTM(self.hidden_size,return_sequences=False))
        model.add(Dropout(self.dropout_rate))
        
        model.add(Flatten()) 
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        if self.weights_path != None:
            model.load_weights(self.weights_path)
        return model

    def make_model_2(self):
        """ 
        Makes the pure LSTM model. This model is: LSTM -> LSTM  -> Dense -> Relu -> Dense -> Softmax.

        :return: None 
        """
        
        model = Sequential()
        model.add(LSTM(
            self.hidden_size,
            input_shape=(self.input_shape[0], self.input_shape[2]),
            recurrent_dropout =  self.dropout_rate,
            return_sequences=True
        ))
        model.add(LSTM(self.hidden_size, return_sequences = False, recurrent_dropout=self.dropout_rate))

        model.add(Dense(self.hidden_size))
        model.add(Activation('relu'))
        
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.vocab_size))
        
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        if self.weights_path != None:
            model.load_weights(self.weights_path)
        return model
        
def train(model, inputs, labels):
    """
    Trains the model. Keras makes thing really simple here so that we don't have to worry too much about batching and saving check points for us. This train function will save the weights per epoch by writing to an hdf5 file.

    :param model: the LSTM + Attention model  
    :param inputs: the input sequences to train the model 
    :param labels: the output "label" for each of the sequences  

    :return: None 
    """

    music_model = model.make_model_1()
    filepath = os.path.abspath("{epoch:03d}-{loss:.2f}.hdf5")
    checkpoint = ModelCheckpoint(
        filepath,
        save_freq = 350, 
        monitor= 'loss',
        verbose = 1,
        save_best_only = False,
        mode= 'min'
    )
    callbacks_list = [checkpoint]
    music_model.fit(inputs, labels, epochs = model.epoch_size, batch_size = model.batch_size, callbacks = callbacks_list, shuffle = True)

