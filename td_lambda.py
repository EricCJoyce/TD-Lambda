import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'							#  Suppress TensorFlow barf.
import shutil
import sys
import time
import tensorflow as tf

from game_environment import GameEnv
from networkagent import NetworkAgent

def main():
	params = get_command_line_params()								#  Collect command-line parameters.
	if params['helpme']:
		usage()
		return

	game = GameEnv()
	agent = NetworkAgent(epsilon=params['epsilon'])

	episode_number = 0

	if not os.path.isdir('./models/'):								#  Create a repository for the models.
		os.mkdir('./models/')

	if os.path.exists('bookmark.txt'):								#  Are we resuming a training session?
		fh = open('bookmark.txt', 'r')
		line = [x for x in fh.readlines() if x[0] != '#']
		fh.close()
		if len(line) > 0:
			arr = line[0].strip().split('\t')
			episode_number = int(arr[0]) + 1						#  The episode number on record is the last one to have been completed.
			if params['verbose']:
				print('\n>>> Resuming with saved model: lodzkaliska-' + arr[0] + '.pb\n')
			agent.model.load_weights('models/lodzkaliska-' + arr[0] + '.pb')
	else:
		fh = open('bookmark.txt', 'w')
		fh.write('#  TD(lambda) training session, started at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') +'\n')
		fh.write('#  python3 ' + ' '.join(sys.argv) + '\n')
		fh.write('#  Line below is the episode number most recently completed.\n')
		fh.close()

	if not os.path.exists('draughts.txt'):
		fh = open('draughts.txt', 'w')
		fh.write('#  TD(lambda) training session, started at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') +'\n')
		fh.write('#  python3 ' + ' '.join(sys.argv) + '\n')
		fh.write('#  Starting position  <t>  Space-separated list of moves  <t>  Result\n')
		fh.close()

	while episode_number < params['episodes']:
		visited_states = {}											#  Blank out the table of visited states.
		visited_states[game.fen] = 1
																	#  Save starting positions for records.
		start_pos = game.fen_lookup_title(game.fen)					#  Save starting position for record.
		history = []												#  Blank out history.

		Alpha = max((1.0 / (1.0 + params['alpha-decay-rate'] * episode_number)) * params['alpha'], params['alpha-min'])
																	#  Start epsilon rot.
		Epsilon = max((1.0 / (1.0 + params['epsilon-decay-rate'] * episode_number)) * params['epsilon'], params['epsilon-min'])
																	#  Start lambda rot.
		Lambda = max((1.0 / (1.0 + params['lambda-decay-rate'] * episode_number)) * params['lambda'], params['lambda-min'])

		if params['verbose']:
			if np.isfinite(params['episodes']):
				print('Episode ' + str(episode_number) + '/' + str(params['episodes']) + ': ' + start_pos)
			else:
				print('Episode ' + str(episode_number) + ': ' + start_pos)
			print('alpha   = ' + str(Alpha))
			print('epsilon = ' + str(Epsilon))
			print('lambda  = ' + str(Lambda))

		x_0, x_1, x_2 = agent.extract_features( game.fen, game )	#  Get features for the side-to-move, from its point of view.
		trace = []													#  Initialize the eligibility trace:
		for trainable in agent.model.trainable_variables:			#  zeroes the same shape as all trainable variables in the model.
			trace.append( np.zeros(trainable.shape) )

		while not game.terminal():
			agent.epsilon = Epsilon									#  (Re)set epsilon.
			move, greedy = agent.select_action(game, visited_states)#  Select an epsilon-greedy move.

			if params['verbose']:
				if greedy:
					print('\t' + game.fen + '\t' + move)
				else:
					print('\t' + game.fen + '\t' + move + ' *')
			history.append(move)

			next_fen = game.make_move(move)
			game.fen = next_fen[:]									#  Update the Game-Environment's state.

			if next_fen not in visited_states:						#  Increment the number of times this state has been visited.
				visited_states[next_fen] = 0
			visited_states[next_fen] += 1

			breakout = False
			for visited_fen, fen_ctr in visited_states.items():		#  If the repeat limit has been reached, abort this episode immediately.
				if fen_ctr > params['repeat-state-limit']:
					record_result = 'Drawn by repetition'
					if params['verbose']:
						print('\t\tEpisode aborted: state repeated too often.')
						game.draw(2)
					breakout = True
					break
			if breakout:
				break

			reward = 0.0											#  Determine reward, if any.
			if game.terminal():										#  Game state is terminal.
				result = game.outcome()								#  Result is in {'l', 'd', '0'}
				if result == 'l':
					record_result = '1-0'
				else:
					record_result = '0-1'
																	#  Move just made was a winning move.
				if (result == 'l' and game.fen[0] == 'd') or (result == 'd' and game.fen[0] == 'l'):
					reward = 1.0
				elif result == '0':									#  Draw.
					reward = 0.0
					record_result = '1/2-1/2'
				else:												#  Move just made was a losing move.
					reward = -1.0

				if params['verbose']:
					if result == 'l':
						print('\t\tLight wins')
					elif result == 'd':
						print('\t\tDark wins')
					else:
						print('\t\tDraw')
					game.draw(2)

			with tf.GradientTape() as tape:							#  Compute v and gradient of v w.r.t. trainable weights in agent.
				v = agent.model( [np.expand_dims(x_0, axis=0), \
				                  np.expand_dims(x_1, axis=0), \
				                  x_2] )							#  (Yes, you have to write it this way.)
			grads = tape.gradient(v, agent.model.trainable_variables)

			flat_grads = []
			for grad in grads:
				flat_grads += list(grad.numpy().flatten('C'))		#  Flatten gradients.
			flat_grads = np.array(flat_grads)
			bogies = list(np.where(np.isinf(flat_grads))[0]) + list(np.where(np.isnan(flat_grads))[0])
			if len(bogies) > 0:										#  Check for bogies.
				print('\nERROR: GRADIENTS HAVE EXPLODED. HALTING.')
				return

			x_next_0, x_next_1, x_next_2 = agent.extract_features( game.fen, game )
			v_next = agent.model.predict( [np.expand_dims(x_next_0, axis=0), \
			                               np.expand_dims(x_next_1, axis=0), \
			                               x_next_2] )[0][0] * -1.0

			for i in range(0, len(grads)):							#  Update the trace.
				trace[i] = params['gamma'] * Lambda * trace[i] + grads[i]
																	#  Compute delta.
			delta = reward + params['gamma'] * v_next - v.numpy()[0][0]

			for i in range(0, len(agent.model.trainable_variables)):
				w = agent.model.trainable_variables[i].numpy()
				w += Alpha * delta * trace[i]
				agent.model.trainable_variables[i].assign(w)

			x_0 = x_next_0											#  Update features.
			x_1 = x_next_1
			x_2 = x_next_2

		game.reset()												#  Reset!

		if episode_number % params['save-every'] == 0:				#  Save the model and the bookmark.
			agent.model.save('./models/lodzkaliska-' + str(episode_number) + '.pb')
			update_bookmark(episode_number)
		update_record(start_pos, history, record_result)			#  Always save the matches. No harm in that.
		episode_number += 1

	return

def update_bookmark(index_completed):
	shutil.copy('bookmark.txt', 'bookmark.backup.txt')

	fh = open('bookmark.txt', 'r')
	header = [x for x in fh.readlines() if x[0] == '#']
	fh.close()

	fh = open('bookmark.txt', 'w')
	for line in header:
		fh.write(line)
	fh.write(str(index_completed) + '\n')
	fh.close()
	return

def update_record(start_pos, history, result):
	shutil.copy('draughts.txt', 'draughts.backup.txt')

	fh = open('draughts.txt', 'r')
	lines = fh.readlines()
	fh.close()

	fh = open('draughts.txt', 'w')
	for line in lines:
		fh.write(line)
	fh.write(start_pos + '\t' + ' '.join(history) + '\t' + result + '\n')
	fh.close()
	return

def get_command_line_params():
	params = {}
	params['episodes'] = float('inf')								#  By default, train forever.
	params['repeat-state-limit'] = 3								#  If any state repeats itself this many times, abort the episode.

	params['alpha'] = 0.0005										#  Alpha is the learning rate. Start high and rot it down.
	params['alpha-decay-rate'] = 0.00005
	params['alpha-min'] = 0.0001

	params['gamma'] = 0.9											#  Gamma is the discounting factor: discount states in the future.

	params['lambda'] = 0.9											#  Lambda is the trace decay parameter. Start high and rot it down.
	params['lambda-decay-rate'] = 0.00005
	params['lambda-min'] = 0.4

	params['epsilon'] = 0.8											#  Epsilon is the number, sampled below which, the agent makes random moves.
	params['epsilon-decay-rate'] = 0.00005							#  Default to a slow decay.
	params['epsilon-min'] = 0.4										#  Never totally lose stochasticity.

	params['batch-size'] = 1										#  Number of simultaneous games to carry on.

	params['save-every'] = 1										#  Save every k* this many matches.

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-episodes', '-repeat', \
	         '-a', '-adecay', '-amin', \
	         '-g', \
	         '-l', '-ldecay', '-lmin', \
	         '-e', '-edecay', '-emin', \
	         '-b', \
	         '-save', \
	         '-v', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-v':
				params['verbose'] = True
			elif sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-episodes':
					params['episodes'] = min(int(argval), 0)
					if params['episodes'] == 0:
						params['episodes'] = float('inf')
				elif argtarget == '-repeat':
					params['repeat-state-limit'] = min(1, int(argval))

				elif argtarget == '-a':
					params['alpha'] = float(argval)
				elif argtarget == '-adecay':
					params['alpha-decay-rate'] = float(argval)
				elif argtarget == '-amin':
					params['alpha-min'] = float(argval)

				elif argtarget == '-g':
					params['gamma'] = max(0.0, min(1.0, float(argval)))

				elif argtarget == '-l':
					params['lambda'] = float(argval)
				elif argtarget == '-ldecay':
					params['lambda-decay-rate'] = float(argval)
				elif argtarget == '-lmin':
					params['lambda-min'] = float(argval)

				elif argtarget == '-e':
					params['epsilon'] = min(max(0.0, float(argval)), 1.0)
				elif argtarget == '-edecay':
					params['epsilon-decay-rate'] = float(argval)
				elif argtarget == '-emin':
					params['epsilon-min'] = min(max(0.0, float(argval)), 1.0)

				elif argtarget == '-b':
					params['batch-size'] = max(1, int(argval))

				elif argtarget == '-save':
					params['save-every'] = max(1, int(argval))

	return params

def usage():
	print('Run TD(lambda) for a deep model in the game INTERNATIONAL DRAUGHTS.')
	print('This script logs all games and saves model versions.')
	print('')
	print('**NOTE**! Not all games make it possible for sequences of moves to do-undo game states. This one does.')
	print('          That is why there is additional code in this version of this script that prohibit *random*')
	print('          moves from returning to any previously visited state.')
	print('          (If all random moves lead to repeat states, then make a greedy choice.)')
	print('          Greedy moves may still repeat states, but that is subject to the repetition limit.')
	print('')
	print('Usage:  python3 td_lambda.py <parameters, preceded by flags>')
	print(' e.g.:  python3 td_lambda.py -repeat 5 -a 0.0001 -g 0.9 -e 0.8 -edecay 0.00005 -emin 0.4 -l 0.8 -ldecay 0.00005 -lmin 0.4 -v')
	print('        This says, "Run forever (until quit). Abort an episode if it repeats any state five times.')
	print('        "Hold the learning rate at 0.0001.')
	print('        "Discount future states by 0.9.')
	print('        "Decay the randomness rate from 0.8 to a minimum of 0.4 at a rate of 0.00005.')
	print('        "Decay the eligibility trace from 0.8 to a minimum of 0.4 at a rate of 0.00005."')
	print('')
	print('Flags:  -episodes   Following int is the number of episodes to run. If this argument is omitted or set to zero,')
	print('                    then training will run forever (until force-quit.)')
	print('        -repeat     Following int is the number of times a game state repeats during a single episode necessary to abort that episode.')
	print('                    Default is 3.')
	print('')
	print('        -a          Following real number is the learning rate. Default is 0.0005.')
	print('        -adecay     Following real number is the learning rate\'s rate of decay. Default is 0.00005.')
	print('        -amin       Following real number is the minimum learning rate. Decay may never lower the learning rate below this limit.')
	print('                    Default is 0.0001.')
	print('                    For each episode, current_alpha = min((1 / (1 + alpha_decay * episode_number)) * alpha, alpha_min).')
	print('')
	print('        -g          Following real number in [0.0, 1.0] is the factor by which we discount future states.')
	print('                    Default is 0.9.')
	print('')
	print('        -l          Following real number is trace factor. Default is 0.9.')
	print('        -ldecay     Following real number is trace\'s rate of decay. Default is 0.00005.')
	print('        -lmin       Following real number is the minimum trace factor. Decay may never lower the influence of the trace below this limit.')
	print('                    Default is 0.4.')
	print('                    For each episode, current_lambda = min((1 / (1 + lambda_decay * episode_number)) * lambda, lambda_min).')
	print('')
	print('        -e          Following real number in [0.0, 1.0] is the number, sampled below which, a random rather than a greedy move is made.')
	print('                    Default is 0.8, which means often random. However, this default depends on also applying epsilon decay.')
	print('        -edecay     Following real number is epsilon\'s rate of decay. Default is 0.00005.')
	print('        -emin       Following real number is the minimum epsilon. Decay may never lower randomness below this limit.')
	print('                    It is a good idea to never completely let go of randomness; maintain an explore-exploit balance.')
	print('                    Default is 0.4.')
	print('                    For each episode, current_epsilon = min((1 / (1 + epsilon_decay * episode_number)) * epsilon, epsilon_min).')
	print('')
	print('        -b          Batch size. For the TD(lambda) algorithm, batch size really means the number of simultaneous games to')
	print('                    carry on at the same time. Obviously some may end before others, so terminated games cease to affect the model.')
	print('                    Default is 1.')
	print('')
	print('        -v          Enable verbosity.')
	print('        -?')
	print('        -help')
	print('        --help      Display this message.')
	return

if __name__ == '__main__':
	main()
