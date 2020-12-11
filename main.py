import sys
import preprocess as preprocess 
import generate as generator 
from model import Model, train

def main():
    """
    Either trains the model or generates a midi file. 
    """
    
    if len(sys.argv) != 2 or sys.argv[1] not in {"TRAIN", "GENERATE"}:
        print("USAGE: python main.py <Action Type>")
        print("<Action Type>: [train/predict]")
        exit()

    print("Preprocessing Data...")

    directory_name = "data"
    notes = preprocess.get_notes(directory_name)
    vocab = preprocess.get_vocab(notes)  
    vocab_size = len(vocab)
    input_windows, label_notes = preprocess.get_windows(notes)

    inputs_g = preprocess.get_inputs_and_labels(input_windows, label_notes, vocab)[0]
    inputs_t, labels_t = preprocess.final_inputs_and_labels(notes, vocab) 

    # Trains the model and saves weights to hdf5 files 
    if sys.argv[1] == "TRAIN":
        print("Training Model...")
        model = Model(inputs_t, vocab_size) 
        train(model, inputs_t, labels_t) 
        print("Finished Training!")

    # Generates MIDI file using some saved weights 
    elif sys.argv[1] == "GENERATE":
        print("Generating MIDI file using loaded weights...")
        model = Model(inputs_t, vocab_size, weights_path = "weights-020-3.7896.hdf5") 
        predicted_notes = generator.generate_notes(model, inputs_g, vocab) 
        generator.generate_midi(predicted_notes) 
        print("Finished generating MIDI!")

if __name__ == '__main__':
    main()