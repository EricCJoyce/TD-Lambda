import numpy as np
import os
import re
import shutil
import struct
import subprocess
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'							#  Suppress TensorFlow barf.

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, concatenate, Lambda, Dropout
from tensorflow.keras.models import load_model

from build_lodzkaliska_model import build_model

def main():
	params = get_command_line_params()								#  Collect parameters.
	if params['helpme']:
		usage()
		return

	if not os.path.exists('./build_neuron_model'):
		print('>>> ERROR: Unable to find executable ./build_neuron_model')
		return

	if not os.path.exists('./models/'):
		print('>>> ERROR: Unable to find directory of *.pb models ./models/')
		return
																	#  Convert to Neuron-C.
	convert_pb_to_nn('models/lodzkaliska-' + str(params['epoch']) + '.pb')

	model = load_model('./models/lodzkaliska-' + str(params['epoch']) + '.pb')
	if not check_pb_nn_outputs(model, 'lodzkaliska-' + str(params['epoch']) + '.nn', params):
		print('>>> ERROR: "lodzkaliska-' + str(params['epoch']) + '.pb" and "lodzkaliska-' + str(params['epoch']) + '.nn" outputs differ more than ' + str(params['epsilon']) + '.')

	return

def check_pb_nn_outputs(model, nn_file, params):
	passed = True
	for fen_side in params['checks']:
		fen = fen_side[0]
		side = fen_side[1]

		args = ['./interpret', '-fen', fen, '-side', side]
		comp_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out = comp_proc.stdout.decode('utf-8')
		err = comp_proc.stderr.decode('utf-8')

		board = [float(x) for x in out.strip().split()]
		tall = np.array([board[:50]]).reshape((10, 5, 1))
		wide = np.array([board[50:100]]).reshape((5, 10, 1))
		toMove = np.array([board[100]]).reshape((1, ))

		y_hat_pb = model.predict( [ np.array([tall]), \
		                            np.array([wide]), \
		                            np.array([toMove]) ] )
		y_hat_pb = y_hat_pb[0][0]

		args = ['./run', nn_file] + [str(x) for x in board]
		comp_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out = comp_proc.stdout.decode('utf-8')
		err = comp_proc.stderr.decode('utf-8')

		y_hat_nn = float(out)

		if params['verbose']:
			print('  >>> ' + str(y_hat_pb) + ' == ' + str(y_hat_nn) + ' +/- ' + str(params['epsilon']))

		passed = passed and abs(y_hat_pb - y_hat_nn) < params['epsilon']

	return passed

def convert_pb_to_nn(network_name):
	conv2dctr = 0
	model = load_model('./' + network_name)
	re.sub('[^0-9]', '', 'network_name')

	dirname = './' + re.sub('[^0-9]', '', network_name) + '/'
	if os.path.exists(dirname):
		shutil.rmtree(dirname)
	os.mkdir(dirname)

	for i in range(0, len(model.layers)):							#  Export Keras weights.

		if isinstance(model.layers[i], keras.layers.Conv2D):
			weights = conv2d_to_weights(model.layers[i])			#  Each in 'weights' is a filter.

			for weightArr in weights:
				fh = open(dirname + 'Conv2D-' + str(conv2dctr) + '.weights', 'wb')
				packstr = '<' + 'd'*len(weightArr)
				fh.write(struct.pack(packstr, *weightArr))
				fh.close()

				conv2dctr += 1

		elif isinstance(model.layers[i], keras.layers.Dense):
			weights = dense_to_weights(model.layers[i])
			layername = model.layers[i].name

			if layername == 'dense256':
				fh = open(dirname + '/Dense-0.weights', 'wb')
			elif layername == 'dense64':
				fh = open(dirname + '/Dense-1.weights', 'wb')
			elif layername == 'dense16':
				fh = open(dirname + '/Dense-2.weights', 'wb')
			elif layername == 'dense1':
				fh = open(dirname + '/Dense-3.weights', 'wb')

			packstr = '<' + 'd'*len(weights)
			fh.write(struct.pack(packstr, *weights))
			fh.close()

																	#  Call build_neuron_model.
	args = ['./build_neuron_model', re.sub('[^0-9]', '', network_name)]
	comp_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out = comp_proc.stdout.decode('utf-8')
	err = comp_proc.stderr.decode('utf-8')

	shutil.rmtree(dirname)

	return

#  Write layer weights to file in ROW-MAJOR ORDER so our C program can read them into the model
def dense_to_weights(layer):
	ret = []

	w = layer.get_weights()
	width = len(w[1])												#  Number of units
	height = len(w[0])												#  Number of inputs (excl. bias)

	for hctr in range(0, height):									#  This is the row-major read
		for wctr in range(0, width):
			ret.append(w[0][hctr][wctr])

	for wctr in range(0, width):
		ret.append(w[1][wctr])

	return ret

#  Return a list of lists of weights.
#  Each can be written to a buffer and passed as weights to a C Conv2DLayer.
def conv2d_to_weights(layer):
	ret = []

	w = layer.get_weights()
	filterW = len(w[0][0])
	filterH = len(w[0])
	numFilters = len(w[1])

	for fctr in range(0, numFilters):
		ret.append( [] )
		for hctr in range(0, filterH):
			for wctr in range(0, filterW):
				ret[-1].append( w[0][hctr][wctr][0][fctr])
		ret[-1].append(w[1][fctr])

	return ret

def get_command_line_params():
	params = {}
	params['epoch'] = None											#  The model to source.
	params['checks'] = [ ('lPPPPPPPPPPPPPPPPPPPP55pppppppppppppppppppp', 'l'), \
	                     ('lPPPPPPPPPPPPPPPPPPPP55pppppppppppppppppppp', 'd'), \
	                     ('dPPPPPPPPPPPPPPPPPP1P3P15pppppppppppppppppppp',  'l'), \
	                     ('dPPPPPPPPPPPPPPPPPP1P3P15pppppppppppppppppppp',  'd'), \
	                     ('lPPPPPPPPPPPPPPPPPP1P3P11p3pp1ppppppppppppppppp',  'l'), \
	                     ('lPPPPPPPPPPPPPPPPPP1P3P11p3pp1ppppppppppppppppp',  'd'), \
	                     ('dPPPPPPPPPPPPPPPPPP1P51p1P1pp1ppppppppppppppppp',  'l'), \
	                     ('dPPPPPPPPPPPPPPPPPP1P51p1P1pp1ppppppppppppppppp',  'd'), \
	                     ('lPPPPPPPPPPPPPPPPPP1P3p11p3pp1p1ppppppppppppppp',  'l'), \
	                     ('lPPPPPPPPPPPPPPPPPP1P3p11p3pp1p1ppppppppppppppp',  'd'), \
	                     ('dPPPPPPPPPPPPPPPPP2P51p1P1pp1p1ppppppppppppppp',  'l'), \
	                     ('dPPPPPPPPPPPPPPPPP2P51p1P1pp1p1ppppppppppppppp',  'd'), \
	                     ('lPPPPPPPPPPPPPPPPP2P4p1p3pp3ppppppppppppppp',  'l'), \
	                     ('lPPPPPPPPPPPPPPPPP2P4p1p3pp3ppppppppppppppp',  'd'), \
	                     ('dPPPPPPPPPPPPPPPPP351p1P1pp3ppppppppppppppp',  'l'), \
	                     ('dPPPPPPPPPPPPPPPPP351p1P1pp3ppppppppppppppp',  'd'), \
	                     ('lPPPPPPPPPPPPPPPPP35pp1P1p4ppppppppppppppp',  'l'), \
	                     ('lPPPPPPPPPPPPPPPPP35pp1P1p4ppppppppppppppp',  'd'), \
	                     ('dPPPPPPPPPPPPPPP1P31P3pp1P1p4ppppppppppppppp',  'l'), \
	                     ('dPPPPPPPPPPPPPPP1P31P3pp1P1p4ppppppppppppppp',  'd'), \
	                     ('lPPPPPPPPPPPPPPPpP35p2P1p4ppppppppppppppp',  'l'), \
	                     ('lPPPPPPPPPPPPPPPpP35p2P1p4ppppppppppppppp',  'd'), \
	                     ('dPPPPPPPPPPP1PPP1P353P1pP3ppppppppppppppp',  'l'), \
	                     ('dPPPPPPPPPPP1PPP1P353P1pP3ppppppppppppppp',  'd'), \
	                     ('lPPPPPPPPPPP1PPP1P35p2P1p4p1ppppppppppppp',  'l'), \
	                     ('lPPPPPPPPPPP1PPP1P35p2P1p4p1ppppppppppppp',  'd'), \
	                     ('dPPPPPPPPPPP1P1P1PP25p2P1p4p1ppppppppppppp',  'l'), \
	                     ('dPPPPPPPPPPP1P1P1PP25p2P1p4p1ppppppppppppp',  'd'), \
	                     ('lPPPPPPPPPPP1P1P1PP25p2P1p4pppppp1pppppppp',  'l'), \
	                     ('lPPPPPPPPPPP1P1P1PP25p2P1p4pppppp1pppppppp',  'd'), \
	                     ('dPPPPPPPPPP2P1PPPP25p2P1p4pppppp1pppppppp',  'l'), \
	                     ('dPPPPPPPPPP2P1PPPP25p2P1p4pppppp1pppppppp',  'd'), \
	                     ('lPPPPPPPPPP2P1PPPP25p2P1p4ppppppppppp1ppp',  'l'), \
	                     ('lPPPPPPPPPP2P1PPPP25p2P1p4ppppppppppp1ppp',  'd'), \
	                     ('dPPPPPPPPPP2P1PPPP25p4p3Pppppppppppp1ppp',  'l'), \
	                     ('dPPPPPPPPPP2P1PPPP25p4p3Pppppppppppp1ppp',  'd'), \
	                     ('lPPPPPPPPPP2P1PPPP25p3pp4ppp1ppppppp1ppp',  'l'), \
	                     ('lPPPPPPPPPP2P1PPPP25p3pp4ppp1ppppppp1ppp',  'd'), \
	                     ('dPPPPPPPPPP2P1P1PP21P3p3pp4ppp1ppppppp1ppp',  'l'), \
	                     ('dPPPPPPPPPP2P1P1PP21P3p3pp4ppp1ppppppp1ppp',  'd'), \
	                     ('lPPPPPPPPPP2P1P1PP21P3p3ppp31pp1ppppppp1ppp',  'l'), \
	                     ('lPPPPPPPPPP2P1P1PP21P3p3ppp31pp1ppppppp1ppp',  'd'), \
	                     ('dPPPPPPPPPP2P1P2P21PP2p3ppp31pp1ppppppp1ppp',  'l'), \
	                     ('dPPPPPPPPPP2P1P2P21PP2p3ppp31pp1ppppppp1ppp',  'd'), \
	                     ('lPPPPPPPPPP2P1P2P252p1ppp31pp1ppppppp1ppp',  'l'), \
	                     ('lPPPPPPPPPP2P1P2P252p1ppp31pp1ppppppp1ppp',  'd'), \
	                     ('dPPPPPPPPPP2P1P52P22p1ppp31pp1ppppppp1ppp',  'l'), \
	                     ('dPPPPPPPPPP2P1P52P22p1ppp31pp1ppppppp1ppp',  'd'), \
	                     ('lPPPPPPPPPP2P1P1p354ppp31pp1ppppppp1ppp',  'l'), \
	                     ('lPPPPPPPPPP2P1P1p354ppp31pp1ppppppp1ppp',  'd'), \
	                     ('dPPPPPPPPPP4P51P34ppp31pp1ppppppp1ppp',  'l'), \
	                     ('dPPPPPPPPPP4P51P34ppp31pp1ppppppp1ppp',  'd'), \
	                     ('lPPPPPPPPPP4P51P34ppp3ppp1pp1pppp1ppp',  'l'), \
	                     ('lPPPPPPPPPP4P51P34ppp3ppp1pp1pppp1ppp',  'd'), \
	                     ('dPPPPPP1PPP1P2P51P34ppp3ppp1pp1pppp1ppp',  'l'), \
	                     ('dPPPPPP1PPP1P2P51P34ppp3ppp1pp1pppp1ppp',  'd'), \
	                     ('lPPPPPP1PPP1P2P51P2p5pp3ppp1pp1pppp1ppp',  'l'), \
	                     ('lPPPPPP1PPP1P2P51P2p5pp3ppp1pp1pppp1ppp',  'd'), \
	                     ('dPPPPPP1PPP1P34P1P2p5pp3ppp1pp1pppp1ppp',  'l'), \
	                     ('dPPPPPP1PPP1P34P1P2p5pp3ppp1pp1pppp1ppp',  'd'), \
	                     ('lPPPPPP1PPP1P34P1P2p5pp2pppp2p1pppp1ppp',  'l'), \
	                     ('lPPPPPP1PPP1P34P1P2p5pp2pppp2p1pppp1ppp',  'd'), \
	                     ('dPPPPPP1PPP1P351P35pp3ppp1Pp1pppp1ppp',  'l'), \
	                     ('dPPPPPP1PPP1P351P35pp3ppp1Pp1pppp1ppp',  'd'), \
	                     ('lPPPPPP1PPP1P351P35pp1p1pp2Pp1pppp1ppp',  'l'), \
	                     ('lPPPPPP1PPP1P351P35pp1p1pp2Pp1pppp1ppp',  'd'), \
	                     ('dPPPPPP1PP11P2P51P35pp1p1pp2Pp1pppp1ppp',  'l'), \
	                     ('dPPPPPP1PP11P2P51P35pp1p1pp2Pp1pppp1ppp',  'd'), \
	                     ('lPPPPPP1PP11P2P51P35pp1p1pp1pPp1pp1p1ppp',  'l'), \
	                     ('lPPPPPP1PP11P2P51P35pp1p1pp1pPp1pp1p1ppp',  'd'), \
	                     ('dPPPPPP1PP11P2P55P4pp1p1pp1pPp1pp1p1ppp',  'l'), \
	                     ('dPPPPPP1PP11P2P55P4pp1p1pp1pPp1pp1p1ppp',  'd'), \
	                     ('lPPPPPP1PP11P2P51p351p1p1pp1pPp1pp1p1ppp',  'l'), \
	                     ('lPPPPPP1PP11P2P51p351p1p1pp1pPp1pp1p1ppp',  'd'), \
	                     ('dPPPPPP1PP11P34P1p351p1p1pp1pPp1pp1p1ppp',  'l'), \
	                     ('dPPPPPP1PP11P34P1p351p1p1pp1pPp1pp1p1ppp',  'd'), \
	                     ('lPPPPPP1PP11P34P1p351p1pppp2Pp1pp1p1ppp',  'l'), \
	                     ('lPPPPPP1PP11P34P1p351p1pppp2Pp1pp1p1ppp',  'd'), \
	                     ('dPPPPPP1PP11P34P1p351p3pp3p2p1pKppp',  'l'), \
	                     ('dPPPPPP1PP11P34P1p351p3pp3p2p1pKppp',  'd'), \
	                     ('lPPPPPP1PP11P34P1p351p3pp3p1pp1pK1pp',  'l'), \
	                     ('lPPPPPP1PP11P34P1p351p3pp3p1pp1pK1pp',  'd'), \
	                     ('dPPPPPP1PP11P31K2P551p31p3p1pp1p2pp',  'l'), \
	                     ('dPPPPPP1PP11P31K2P551p31p3p1pp1p2pp',  'd'), \
	                     ('lPPPPPP1PP11P31K2P51p351p3p1pp1p2pp',  'l'), \
	                     ('lPPPPPP1PP11P31K2P51p351p3p1pp1p2pp',  'd'), \
	                     ('dPPPPPP1PP11PK24P51p351p3p1pp1p2pp',  'l'), \
	                     ('dPPPPPP1PP11PK24P51p351p3p1pp1p2pp',  'd'), \
	                     ('lPPPPPP1PP11PK24P51p351p3p1pppp2p1',  'l'), \
	                     ('lPPPPPP1PP11PK24P51p351p3p1pppp2p1',  'd'), \
	                     ('dPPPPPPKPP11P34P51p351p3p1pppp2p1',  'l'), \
	                     ('dPPPPPPKPP11P34P51p351p3p1pppp2p1',  'd'), \
	                     ('lPPPPPPKPP11P34P51p351p3ppppp3p1',  'l'), \
	                     ('lPPPPPPKPP11P34P51p351p3ppppp3p1',  'd'), \
	                     ('dPPPPPPKPP151P2P51p351p3ppppp3p1',  'l'), \
	                     ('dPPPPPPKPP151P2P51p351p3ppppp3p1',  'd'), \
	                     ('lPPPPPPKPP151P2P51p31p35ppppp3p1',  'l'), \
	                     ('lPPPPPPKPP151P2P51p31p35ppppp3p1',  'd'), \
	                     ('dPPPPPP1PP151PK1P51p31p35ppppp3p1',  'l'), \
	                     ('dPPPPPP1PP151PK1P51p31p35ppppp3p1',  'd'), \
	                     ('lPPPPPP1PP151PK1P51p31p31p3pp1pp3p1',  'l'), \
	                     ('lPPPPPP1PP151PK1P51p31p31p3pp1pp3p1',  'd'), \
	                     ('dPPPPPP1PP151P2P51p31p2K1p3pp1pp3p1',  'l'), \
	                     ('dPPPPPP1PP151P2P51p31p2K1p3pp1pp3p1',  'd'), \
	                     ('lPPPPPP1PP151P2P51p31p2Kpp3p2pp3p1',  'l'), \
	                     ('lPPPPPP1PP151P2P51p31p2Kpp3p2pp3p1',  'd'), \
	                     ('dPPPPPP1PP151P2P51p31p3pp3p3p2Kp1',  'l'), \
	                     ('dPPPPPP1PP151P2P51p31p3pp3p3p2Kp1',  'd'), \
	                     ('lPPPPPP1PP151P2P51p31p3pp2pp42Kp1',  'l'), \
	                     ('lPPPPPP1PP151P2P51p31p3pp2pp42Kp1',  'd'), \
	                     ('dPPPPPP1PP151P2P51p2K1p3pp2pp43p1',  'l'), \
	                     ('dPPPPPP1PP151P2P51p2K1p3pp2pp43p1',  'd'), \
	                     ('lPPPPPP1PP151P2P51p2K1p3pp2pp3p5',  'l'), \
	                     ('lPPPPPP1PP151P2P51p2K1p3pp2pp3p5',  'd'), \
	                     ('dPPPPPP1PP154P2P21p2K1p3pp2pp3p5',  'l'), \
	                     ('dPPPPPP1PP154P2P21p2K1p3pp2pp3p5',  'd'), \
	                     ('lPPPPPP1PP152p1P54K1p3pp2pp3p5',  'l'), \
	                     ('lPPPPPP1PP152p1P54K1p3pp2pp3p5',  'd'), \
	                     ('dPP1PPPPPP152p1P54K1p3pp2pp3p5',  'l'), \
	                     ('dPP1PPPPPP152p1P54K1p3pp2pp3p5',  'd'), \
	                     ('lPP1PPPPPP152p1P54K1pp2p3pp3p5',  'l'), \
	                     ('lPP1PPPPPP152p1P54K1pp2p3pp3p5',  'd'), \
	                     ('dPP1PPPPPP152pKP551pp2p3pp3p5',  'l'), \
	                     ('dPP1PPPPPP152pKP551pp2p3pp3p5',  'd'), \
	                     ('lPP1PPPPPP152pKP52p21p3p3pp3p5',  'l'), \
	                     ('lPP1PPPPPP152pKP52p21p3p3pp3p5',  'd'), \
	                     ('dPP1PPPPPP152p1PK455p3pp3p5',  'l'), \
	                     ('dPP1PPPPPP152p1PK455p3pp3p5',  'd'), \
	                     ('lPP1PPPPPP152p1PK454pp4p3p5',  'l'), \
	                     ('lPP1PPPPPP152p1PK454pp4p3p5',  'd'), \
	                     ('dPP1PPPPPP15K1p1P554pp4p3p5',  'l'), \
	                     ('dPP1PPPPPP15K1p1P554pp4p3p5',  'd'), \
	                     ('lPP1PPPPPP13p1K3P554pp4p3p5',  'l'), \
	                     ('lPP1PPPPPP13p1K3P554pp4p3p5',  'd'), \
	                     ('dPP1PPPP1P15K2PP554pp4p3p5',  'l'), \
	                     ('dPP1PPPP1P15K2PP554pp4p3p5',  'd'), \
	                     ('lPP1PPPP1P15K2PP554pp2p1p45',  'l'), \
	                     ('lPP1PPPP1P15K2PP554pp2p1p45',  'd'), \
	                     ('dPP1PPPP1P153PP1K354pp2p1p45',  'l'), \
	                     ('dPP1PPPP1P153PP1K354pp2p1p45',  'd'), \
	                     ('lPP1PPPP1P153PP1K353ppp4p45',  'l'), \
	                     ('lPP1PPPP1P153PP1K353ppp4p45',  'd'), \
	                     ('dPP1PPPP1P15K2PP553ppp4p45',  'l'), \
	                     ('dPP1PPPP1P15K2PP553ppp4p45',  'd'), \
	                     ('lPP1PPPP1P15K2PP55p2pp5p45',  'l'), \
	                     ('lPP1PPPP1P15K2PP55p2pp5p45',  'd'), \
	                     ('dPP1PPPP1P153PP51K3p2pp5p45',  'l'), \
	                     ('dPP1PPPP1P153PP51K3p2pp5p45',  'd'), \
	                     ('lPP1PPPP1P153PP51K3p2ppp455',  'l'), \
	                     ('lPP1PPPP1P153PP51K3p2ppp455',  'd'), \
	                     ('dPP1PPPP1P153PP55p2pp5K45',  'l'), \
	                     ('dPP1PPPP1P153PP55p2pp5K45',  'd'), \
	                     ('lPP1PPPP1P153PP53p1p2p15K45',  'l'), \
	                     ('lPP1PPPP1P153PP53p1p2p15K45',  'd'), \
	                     ('dPP1PPPP1P153PP53p1p2p155K4',  'l'), \
	                     ('dPP1PPPP1P153PP53p1p2p155K4',  'd'), \
	                     ('lPP1PPPP1P153PP52pp1p455K4',  'l'), \
	                     ('lPP1PPPP1P153PP52pp1p455K4',  'd'), \
	                     ('dPP1PPPP1P153PP55p44K55',  'l'), \
	                     ('dPP1PPPP1P153PP55p44K55',  'd'), \
	                     ('lPP1PPPP1P153PP5p454K55',  'l'), \
	                     ('lPP1PPPP1P153PP5p454K55',  'd'), \
	                     ('dPP1PPPP1P153PP3K1p45555',  'l'), \
	                     ('dPP1PPPP1P153PP3K1p45555',  'd'), \
	                     ('lPP1PPPP1P153PPp2K155555',  'l'), \
	                     ('lPP1PPPP1P153PPp2K155555',  'd'), \
	                     ('dPP1PP1P1P11P33PPp2K155555',  'l'), \
	                     ('dPP1PP1P1P11P33PPp2K155555',  'd'), \
	                     ('lPP1PP1P1P11P3p2PP3K155555',  'l'), \
	                     ('lPP1PP1P1P11P3p2PP3K155555',  'd'), \
	                     ('dPP1PP1P1P153PPP2K155555',  'l'), \
	                     ('dPP1PP1P1P153PPP2K155555',  'd') ]
	params['epsilon'] = 0.00001

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-v', '-?', '-help', '--help']

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
				if argtarget == '-e':
					params['epoch'] = int(argval)					#  Keep it above 0
	return params

def usage():
	print('Convert a specified *.pb model to an *.nn model.')
	print('')
	print('Usage:  python3 convert_to_nn.py <parameters, preceded by flags>')
	print(' e.g.:  python3 convert_to_nn.py -e 0 -v')
	print('')
	print('Flags:  -e   The target number. Given this, the script will expect to find "./models/lodzkaliska-<e>.pb".')
	print('')
	print('        -v   Enable verbosity.')
	print('        -?   Display this message.')
	return

if __name__ == '__main__':
	main()