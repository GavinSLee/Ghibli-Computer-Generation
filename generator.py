import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data, get_batch


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class is used to generate Ghibli Music using LSTM

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # hyperparameters

        self.vocab_size = vocab_size
        self.window_size = 20 
        self.embedding_size = 50
        self.batch_size = 300 
        self.hidden_size = 1000
        self.learning_rate = 0.005
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=.1))
        self.lstm = tf.keras.layers.LSTM(self.embedding_size, return_sequences=True, return_state=True)
        self.dense1 = tf.keras.layers.Dense(self.hidden_size, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(self.vocab_size, activation = 'softmax')


    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """
        
        #lookup embedding
        embedding = tf.nn.embedding_lookup(self.E, inputs); 

        #run through lstm
        whole_seq_output, final_memory_state, final_carry_state = self.lstm(embedding, initial_state = initial_state)

        #two linear layers (first one has Relu, second one has softmax)
        dense1_output = self.dense1(whole_seq_output)
        probabilities = self.dense2(dense1_output)
        return probabilities, (final_memory_state, final_carry_state)

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        #using sparse categorical cross entropy
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs, axis=-1)
        avgLoss = tf.math.reduce_mean(loss) #average loss recorded
        return avgLoss


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    #same windows_count for both inputs and labels
    windows_count = len(train_inputs) // model.window_size
    #remove excess elements that don't fit the window
    train_inputs = train_inputs[:windows_count * model.window_size]
    train_labels = train_labels[:windows_count * model.window_size]
    #reshaping 
    train_inputs = train_inputs.reshape(-1, model.window_size)
    train_labels = train_labels.reshape(-1, model.window_size)

    #count how many batches are being made
    batch_count = train_inputs.shape[0] // model.batch_size
    for i in range(batch_count):
        #get batch / slices from the input and labels
        batch_inputs, batch_labels = get_batch(train_inputs, train_labels, i * model.batch_size, model.batch_size)
        with tf.GradientTape() as tape: #gradient descent
            probabilities = model.call(batch_inputs, None)[0]
            loss = model.loss(probabilities, batch_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    
    #same windows_count for both inputs and labels
    windows_count = len(test_inputs) // model.window_size
    #remove excess elements that don't fit the window
    test_inputs = test_inputs[:windows_count * model.window_size]
    test_labels = test_labels[:windows_count * model.window_size]
    #reshaping 
    test_inputs = test_inputs.reshape(-1, model.window_size)
    test_labels = test_labels.reshape(-1, model.window_size)

    #count how many batches are being made
    batch_count = test_inputs.shape[0] // model.batch_size
    losses_sum = 0; #sum up all the losses and average it here
    for i in range(batch_count):
        batch_inputs, batch_labels = get_batch(test_inputs, test_labels, i * model.batch_size, model.batch_size)
        probabilities = model.call(batch_inputs, None)[0]
        losses_sum += model.loss(probabilities, batch_labels)
    avglosses = losses_sum / batch_count
    return np.e**avglosses #perplexity
