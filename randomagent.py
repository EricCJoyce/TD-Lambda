'''
For INTERNATIONAL DRAUGHTS.
'''
import numpy as np

class RandomAgent(object):
	def __init__(self):
		'''
		Nothing. Nothing to do.
		'''

	def select_random_move(self, game):
		moves = game.moves()
		return np.random.choice(moves)
