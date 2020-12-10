# Glorious-Ghibli-Generation

<p align = "center">
 
 <img src = "https://user-images.githubusercontent.com/46236119/101830644-b0a00480-3ae9-11eb-9537-2dd9ab032040.jpg" />
  
</p>

## Introduction
This is a final project completed for Brown University's Deep Learning Course (CSCI 1470). The project was completed by Andrew Kim (akim112) and Gavin Lee (glee56). This README will be rather dense, as the full write-up can be viewed here: 

https://docs.google.com/document/d/1i0tiem_6-IsJXpxYPN604SRDo_zywTPekJL1cOsL8MM/edit?usp=sharing

This README will mostly contain info on how to train our model and generate a sample MIDI output. 

## Virtual Environment

First, you need to setup a virtual environment. Run the following command in your terminal to setup the virtual environment:

```
bash create_venv.sh
```

Next, to activate the virtual environment, run the following:

```
source ./env/bin/activate
```

## Training the Model and Generating a MIDI File 

To train the model, run the following in a terminal:

```
python main.py TRAIN
```

To generate a MIDI file, run the following in a terminal:

```
python main.py GENERATE
```







