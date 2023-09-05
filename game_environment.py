'''
For INTERNATIONAL DRAUGHTS.
Expects:
	- ./makemove
	- ./teammoves
	- ./draw
	- ./victory
'''
import numpy as np
import subprocess

class GameEnv(object):
	def __init__(self, fen=None):
		self.fen = fen
		if self.fen is None:
			self.reset()

	def reset(self):
		self.fen = 'lPPPPPPPPPPPPPPPPPPPP55pppppppppppppppppppp'

	def now_to_move(self):
		return self.fen[0]

	#  Call the C program to apply the given move (a string with hyphens) to 'fen'.
	#  If 'fen' is omitted, then use this object's current self.fen--BUT DO NOT CHANGE THE VALUE OF self.fen!!
	#  Retutn the FEN returned by the C program.
	def make_move(self, move, fen=None):
		if fen is None:
			use_this_fen = self.fen[:]
		else:
			use_this_fen = fen

		if move is None:
			return use_this_fen

		args = ['./makemove', '-fen', use_this_fen, '-m'] + move.split('-')
		comp_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out = comp_proc.stdout.decode('utf-8')
		err = comp_proc.stderr.decode('utf-8')

		assert len(out) > 0, 'ERROR: make_move(' + use_this_fen + ', ' + move + ') failed.'

		return out

	#  Call the C program to get a list of moves for the side to move in fen.
	#  Return a list of hyphenated strings.
	def moves(self):
		args = ['./teammoves', '-fen', self.fen]
		comp_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out = comp_proc.stdout.decode('utf-8')
		err = comp_proc.stderr.decode('utf-8')

		assert len(out) > 0, 'ERROR: team_moves(' + self.fen + ') failed.'

		arr = out.split('|')
		ret = []
		for i in range(2, len(arr)):								#  Skip part the game label and the query-FEN.
			x = arr[i].split('/')									#  Separate substrings.
			ret.append( '-'.join(x[:-1]) )							#  Re-assemble the move elements with a hyphen. Leave off the yielded FEN.

		return ret

	#  Is the current FEN a terminal state? Return True or False.
	def terminal(self, fen=None):
		if fen is None:
			use_this_fen = self.fen[:]
		else:
			use_this_fen = fen
		args = ['./victory', use_this_fen]
		comp_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out = comp_proc.stdout.decode('utf-8')
		err = comp_proc.stderr.decode('utf-8')

		assert len(out) > 0, 'ERROR: terminal(' + use_this_fen + ') failed.'
		arr = out.split('|')

		return arr[1] == 'true'

	#  If the current FEN is a terminal state, then what is the result? In {'l', 'd', '0'}.
	#  If the current FEN is NOT a terminal state, then return False.
	def outcome(self, fen=None):
		if fen is None:
			use_this_fen = self.fen[:]
		else:
			use_this_fen = fen
		args = ['./victory', use_this_fen]
		comp_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out = comp_proc.stdout.decode('utf-8')
		err = comp_proc.stderr.decode('utf-8')

		assert len(out) > 0, 'ERROR: outcome(' + use_this_fen + ') failed.'
		arr = out.split('|')

		if arr[1] == 'true':
			return arr[2]

		return False

	def draw(self, indent=0):
		args = ['./draw', self.fen]
		comp_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out = comp_proc.stdout.decode('utf-8')
		err = comp_proc.stderr.decode('utf-8')

		assert len(out) > 0, 'ERROR: draw(' + self.fen + ') failed.'

		out = out.split('\n')
		for line in out:
			print('\t'*indent + line)
		return

	def fen_lookup_title(self, fen):
		if fen == 'lPPPPPPPPPPPPPPPPPPPP55pppppppppppppppppppp':
			return 'Draughts'
		return None