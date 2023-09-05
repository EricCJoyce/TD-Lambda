'''
For INTERNATIONAL DRAUGHTS.
Expects:
	- ./interpret
'''
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'							#  Suppress TensorFlow barf.
import subprocess
import tensorflow as tf
from build_lodzkaliska_model import build_model						#  Import the script built by the script.

class NetworkAgent(object):
	def __init__(self, **kwargs):
		self.model = build_model()

		if 'epsilon' in kwargs:
			assert isinstance(kwargs['epsilon'], float) and kwargs['epsilon'] >= 0.0 and kwargs['epsilon'] <= 1.0, 'Argument \'epsilon\' passed to NetworkAgent() must be a real number in [0.0, 1.0].'
			self.epsilon = kwargs['epsilon']
		if 'verbose' in kwargs:
			assert isinstance(kwargs['verbose'], bool), 'Argument \'verbose\' passed to NetworkAgent() must be a Boolean.'
			if kwargs['verbose']:
				self.model.summary()

	#  INTERNATIONAL DRAUGHTS makes it possible that a sequence of moves may do-undo game states.
	#  Random moves should never facilitate this, so this method has been updated to include an optional dictionary of all previous states.
	def select_action(self, game, previous_states=None):
		greedy = np.random.sample() >= self.epsilon

		all_moves = game.moves()									#  Compute all moves, all child-states.
		all_child_fens = [game.make_move(move) for move in all_moves]
		if previous_states is not None:								#  Were we given previous states? Consult them.
			permissible_moves = [all_moves[i] for i in range(0, len(all_moves)) if all_child_fens[i] not in previous_states]

		if greedy:													#  Make a greedy choice.
			move = self.select_greedy_move(game)					#  Select move preferred by the current weights.
																	#  Given previous states: consult them.
			if previous_states is not None and move not in permissible_moves and len(permissible_moves) > 0:
				move = np.random.choice(permissible_moves)			#  If the greedy move creates a repetition,
				greedy = False										#  then pick randomly from among the permissible moves.
		else:														#  Make a random choice.
			move = self.select_random_move(game)

			if previous_states is not None and move not in permissible_moves and len(permissible_moves) > 0:
				move = np.random.choice(permissible_moves)			#  If the greedy move creates a repetition,
																	#  then pick randomly from among the permissible moves.
		return move, greedy

	def select_greedy_move(self, game):
		moves = game.moves()										#  List of strings representing moves available from the current state.
		X_0 = np.zeros((len(moves), 10, 5))							#  Prepare a batch of feature tensors.
		X_1 = np.zeros((len(moves), 5, 10))
		X_2 = np.zeros((len(moves), 1, 1))
		v = np.zeros((len(moves), 1))								#  Prepare a receptical for evaluations.
		for i in range(0, len(moves)):
			child_fen = game.make_move(moves[i])					#  Make a new FEN.

			if game.terminal(child_fen):							#  Is it terminal?
				result = game.outcome(child_fen)
				if result == child_fen[0]:							#  'child_fen' is a win for the side NOW to move.
					v[i][0] = 1.0
				elif result == '0':									#  'child_fen' ends the game in a draw.
					v[i][0] = 0.0
				else:												#  'child_fen' plays into a loss for the side NOW to move.
					v[i][0] = -1.0
			else:
				X_0[i], X_1[i], X_2[i] = self.extract_features(child_fen, game)

		v += self.model.predict([X_0, X_1, X_2]) * -1.0
																	#  Have the model evaluate ALL the new, subsequent states as a BATCH. It's FASTER!
																	#  And then ADD these values to 'v' so we won't overwrite the rewards for terminal states.
																	#  ALWAYS negate the evaluations made by the opposition.
		value_best = float('-inf')
		action_best = None
		for i in range(0, len(moves)):
			if v[i][0] > value_best:								#  Better value?
				value_best = v[i][0]
				action_best = moves[i][:]

		return action_best

	def select_random_move(self, game):
		moves = game.moves()
		return np.random.choice(moves)

	#  Return THREE NUMPY ARRAYS, shaped (10, 5), (5, 10), and (1, ).
	#                                    (h, w)   (h, w)
	def extract_features(self, fen, game, evaluating_team=None):
		if game.terminal(fen):
			return self.terminal_features()

		if evaluating_team == 'Light':
			args = ['./interpret', '-fen', fen, '-side', 'l']
		elif evaluating_team == 'Dark':
			args = ['./interpret', '-fen', fen, '-side', 'd']
		else:
			evaluating_team = fen[0]
			args = ['./interpret', '-fen', fen, '-side', evaluating_team]

		comp_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out = comp_proc.stdout.decode('utf-8')
		err = comp_proc.stderr.decode('utf-8')

		assert len(out) > 0, 'ERROR: extract_features(' + fen + ', ' + evaluating_team + ') failed.'
		arr = [float(x) for x in out.split()]						#  Not normalized or anything. Straight out the C code (snip final sum).
		return np.array( arr[  :50 ] ).reshape((10, 5)), \
		       np.array( arr[50:100] ).reshape((5, 10)), \
		       np.array( arr[100] ).reshape((1, ))

	#  Feature vectors for terminal states are zero vectors.
	def terminal_features(self):
		return np.zeros((10, 5), np.float32), np.zeros((5, 10), np.float32), np.zeros((1, ), np.float32)
