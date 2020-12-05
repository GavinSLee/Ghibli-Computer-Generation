from lstm import get_notes, prepare_sequences, create_network, train

import tensorflow as tf




if __name__ == "__main__":    
    #load files in
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    #train
    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    model.fit(network_input, network_output, epochs=1, batch_size=128, shuffle=True)

    model.load_weights("weights-060-4.7109.hdf5")

    train(model, network_input, network_output)

