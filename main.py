import sys
from preprocess import get_notes, prepare_sequences
from predict import generate
from model import Model, train

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"train", "generate"}:
        print("USAGE: python main.py <Action Type>")
        print("<Action Type>: [train/predict]")
        exit()

    # Initialize model
    if sys.argv[1] == "train":
        notes = get_notes()

        # get amount of pitch names
        n_vocab = len(set(notes))

        #prepare training inputs and labels
        network_input, network_output = prepare_sequences(notes, n_vocab)
        model = Model(network_input, n_vocab)

        train(model, network_input, network_output)

    elif sys.argv[1] == "generate":
        generate()


if __name__ == '__main__':
    main()