# TD-Lambda
## Train a deep network agent using TD(lambda)
This code was developed to train "&#321;&#243;d&#378; Kaliska," a deep network to evaluate states in International Draughts. This code can be easily modified to work in other contexts.

### Build the code that builds your model
Sticking with "&#321;&#243;d&#378; Kaliska" for the sake of explanation, your first step is to run `python build_model_code.py`. This script generates `build_neuron_model.c` and `build_lodzkaliska_model.py` according to a modifiable network structure. The basic, two-branch design of the network is assumed, but `build_model_code.py` consults `Conv-filter-map.txt` to determine the shapes and quantities of convolutional kernels. Modify this text file to modify the network built by this script.

### Compile game sub-programs
Game code for Draughts was written in C and is not included in this repository. The Python scripts and classes call compiled programs and collect their standard outputs. This was done so that I could use TD-Lambda training for any game agent written in any language. The Python code in this repository expects the following programs:
- `./draw`: Display the state of the given board.
- `./interpret`: Convert the given Forsyth-Edwards Notation (FEN) string into the appropriate array of values for input to a neural network. Print these values to standard output.
- `./makemove`: Apply the given move to the given game state and print a FEN string for the updated game state.
- `./run`: Pass the given input to the given neural network file and print the network's output.
- `./teammoves`: Given a FEN string and an indicator of which side to move, print a separator-delimited string of available moves.
- `./victory`: Given a FEN string, print an indication of which side has won the game, or an indication that the game is not yet over.

### Train
Chosing training parameters is highly case-specific. Expect to scrap and restart several training sessions as you zero in on parameters that help your model to be its best. Also expect progress to emerge slowly after tens of thousands of episodes.

`python td_lambda.py -repeat 5 -a 0.0001 -g 0.9 -e 0.8 -edecay 0.00005 -emin 0.4 -l 0.8 -ldecay 0.00005 -lmin 0.4 -v`

This script call:
- Will abandon an episode if a game state is repeated five times.
- Uses a fixed learning rate of 0.0001.
- Uses a discount (`g`) of 0.9.
- Starts epsilon (`e`) at 0.8, decays it per episode by a rate of 0.00005, and clamps a minimum of 0.4.
- Starts lambda (`l`) at 0.8, decays it per episode by a rate of 0.00005, and clamps a minimum of 0.4.
- Runs in verbose mode, showing the play by play.
