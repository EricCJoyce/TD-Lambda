# TD-Lambda
Train a deep network agent using TD(lambda)

This code was developed to train a deep network to evaluate states in International Draughts, but it can easily be modified to work in other contexts.

Game code was written in C and is not included in this repository. The Python scripts and classes call compiled programs and collect their standard outputs. This was done so that I could use TD-Lambda training for any game agent written in any language. The Python code in this repository expects the following programs:
- `draw`: Display the state of the given board.
- `interpret`: Convert the given Forsyth-Edwards Notation (FEN) string into the appropriate array of values for input to a neural network. Print these values to standard output.
- `makemove`: Apply the given move to the given game state and print a FEN string for the updated game state.
- `run`: Pass the given input to the given neural network file and print the network's output.
- `teammoves`: Given a FEN string and an indicator of which side to move, print a separator-delimited string of available moves.
- `victory`: Given a FEN string, print an indication of which side has won the game, or an indication that the game is not yet over.
