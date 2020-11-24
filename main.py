from generator import Model, train, test
from preprocess import get_data, get_batch


def main():
    # TO-DO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see impossibly small perplexities.
    
    #get train and test tokens with vocab dict from preprocessing
    train_tok, test_tok, vocab_dict = get_data("../../data/train.txt", "../../data/test.txt")

    #"shifting" the inputs and labels so that it fits the RNN problem
    train_inputs = train_tok[:-1]
    train_labels = train_tok[1:]
    test_inputs = test_tok[:-1]
    test_labels = test_tok[1:]

    # initialize model and tensorflow variables
    model = Model(len(vocab_dict))

    # training
    train(model, train_inputs, train_labels)

    # testing 
    perplexity = test(model, test_inputs, test_labels)

    # Print out perplexity 
    print(perplexity)

if __name__ == '__main__':
    main()