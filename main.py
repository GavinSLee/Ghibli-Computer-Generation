import sys
import preprocess as preprocess 
import generate as generator 
from model import Model, train

def main():
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
    inputs_g, labels_g = preprocess.get_inputs_and_labels(input_windows, label_notes, vocab)  
    
    inputs, labels = preprocess.final_inputs_and_labels(notes, vocab) 

    # Initialize model
    if sys.argv[1] == "TRAIN":
        print("Training Model...")
        model = Model(inputs, vocab_size) 
        train(model, inputs, labels) 
        print("Finished Training!")

    # Generates MIDI file using some saved weights 
    elif sys.argv[1] == "GENERATE":
        print("Generating MIDI file using loaded weights...")
        model = Model(inputs, vocab_size) 
        music_model = model.make_model("weights.hdf5")

        predicted_notes = generator.generate_notes(music_model, inputs_g, vocab) 
        generator.generate_midi(predicted_notes) 
        print("Finished generating MIDI!")

if __name__ == '__main__':
    main()