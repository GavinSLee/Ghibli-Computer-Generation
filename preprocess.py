import tensorflow as tf
import numpy as np
from functools import reduce


def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """
    # load and concatenate training data from training file.
    train_tok = []
    with open(train_file, 'r') as f:
        train_tok = f.read().strip().split()

    # load and concatenate testing data from testing file.
    test_tok = []
    with open(test_file, 'r') as f:
        test_tok = f.read().strip().split()

    # read in and tokenize training data (I also make the vocab_dict while tokenizing train_tok)
    vocab_dict = {}
    for i in range(len(train_tok)):
        tok = train_tok[i]
        if not tok in vocab_dict:
            vocab_dict[tok] = len(vocab_dict)
        train_tok[i] = vocab_dict.get(tok)
    
    # read in and tokenize testing data
    for i in range(len(test_tok)):
        tok = test_tok[i]
        # Ensure that all words appearing in test also appear in train
        assert tok in vocab_dict            
        test_tok[i] = vocab_dict.get(tok)

    # TODO: return tuple of training tokens, testing tokens, and the vocab dictionary.
    return (np.array(train_tok), np.array(test_tok), vocab_dict)

def get_batch(inputs, labels, start_index, batch_size):
    """
	useful method that slices the inputs and labels and returns the snippet of the data from 
	start_index to start_index + batch_size. The return data has a size of batch_size, which is 
	then used in train method for batching. 
	:param inputs: numpy list of inputs
	:param labels: numpy list of labels
	:param start_index: starting index of the batch. 
	:param batch_size: size of your batch
	:return: tuple of inputs and labels, each with size of batch_size
	"""
    return (inputs[start_index:start_index+batch_size], labels[start_index:start_index+batch_size])