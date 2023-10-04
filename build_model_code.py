import sys

def main():
	params = get_command_line_params()								#  Collect parameters.
	if params['helpme']:
		usage()
		return

	build_model_code(params)

	return

def build_model_code(params):
	conv_map = {}													#  key:(barrage, (filter-shape)) ==> val: quantity
	fh = open('Conv-filter-map.txt', 'r')
	for line in fh.readlines():
		if line[0] != '#' and len(line) > 1:
			arr = line.strip().split('\t')
			barrage = arr[0]
			filter_shape = tuple([int(x) for x in arr[1].split()])
			quantity = int(arr[2])

			conv_map[ (barrage, filter_shape) ] = quantity
	fh.close()

	fh_py = open('build_lodzkaliska_model.py', 'w')					#  Automatically generate explicit Python code to build the network.
	fh_py.write('import tensorflow as tf\n')
	fh_py.write('import tensorflow.keras as keras\n')
	fh_py.write('from tensorflow.keras import models\n')
	fh_py.write('from tensorflow.keras import optimizers\n')
	fh_py.write('from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, concatenate, Lambda, Dropout\n')
	fh_py.write('from tensorflow.keras.callbacks import ModelCheckpoint\n')
	fh_py.write('\n')
	fh_py.write('def build_model():\n')
	fh_py.write('\ttallEncode = Input(shape=(10, 5, 1))\n')
	fh_py.write('\twideEncode = Input(shape=(5, 10, 1))\n')
	fh_py.write('\tmoveEncode = Input(shape=(1, ))\n')

	fh_c = open('build_neuron_model.c', 'w')						#  Automatically generate C code to build the Neuron-C network.
	fh_c.write('#include "neuron.h"\n')
	fh_c.write('\n')
	fh_c.write('unsigned int readWeights(char*, double**);\n')
	fh_c.write('\n')
	fh_c.write('int main(int argc, char* argv[])\n')
	fh_c.write('  {\n')
	fh_c.write('    NeuralNet* nn;\n')
	fh_c.write('    double* w = NULL;\n')
	fh_c.write('    char buffer[256];\n')
	fh_c.write('    unsigned char len;\n')
	fh_c.write('    unsigned char i;\n')
	fh_c.write('\n')
	fh_c.write('    init_NN(&nn, 101);                                              //  Initialize for input 101-vec\n')
	fh_c.write('\n')
	fh_c.write('    /******************************************************************************/\n')
	fh_c.write('    /***************************************************************    C O N V 2 */\n')
	fh_c.write('    add_Conv2D(5, 10, nn);                                          //  Add a Conv2D layer that receives the TALL input: 5 x 10\n')
	fh_c.write('    setName_Conv2D("Conv2D-Tall", nn->convlayers);                  //  Name the Conv2D layer\n')
	fh_c.write('\n')

	#################################################################  The Tall barrage
																	#  First, sort Tall by width, then by height
	tall_items = sorted([x for x in conv_map.items() if x[0][0] == 'Tall'], key=lambda x: (x[0][0], x[0][1][0]))

	tall_filter_counter = 0
	tall_barrage_output_length = 0

	for k, v in tall_items:											#  Select operations by width
		quantity = v
		filter_w = k[1][0]
		filter_h = k[1][1]

		fh_py.write('\tconvTall'+str(filter_w)+'x'+str(filter_h)+' = Conv2D('+str(quantity)+', ('+str(filter_h)+', '+str(filter_w)+'), activation=\'relu\', padding=\'valid\', input_shape=(10, 5, 1), name=\'convTall'+str(filter_w)+'x'+str(filter_h)+'\')(tallEncode)\n')
		fh_py.write('\tflTall'+str(filter_w)+'x'+str(filter_h)+' = Flatten(name=\'flTall'+str(filter_w)+'x'+str(filter_h)+'\')(convTall'+str(filter_w)+'x'+str(filter_h)+')\n')

		if quantity > 1:											#  Lambda layers are for interleaving; don't bother if we only have one kernel per shape.
			for i in range(0, quantity):
				fh_py.write('\tlambdaTall'+str(filter_w)+'x'+str(filter_h)+'_'+str(i)+' = Lambda(lambda x: x[:, '+str(i)+'::'+str(quantity)+'], name=\'lambdaTall'+str(filter_w)+'x'+str(filter_h)+'_'+str(i)+'\')(flTall'+str(filter_w)+'x'+str(filter_h)+')\n')

		tall_barrage_output_length += output_shape('Tall', filter_w, filter_h, quantity)

		for q in range(0, quantity):
			fh_c.write('    add_Conv2DFilter(' + str(filter_w) + ', ' + str(filter_h) + ', nn->convlayers);                         //  ')
			if q == 0:
				fh_c.write('Add ' + str(quantity) + ' (' + str(filter_w) + ' x ' + str(filter_h) + ') kernels: ')
			else:
				fh_c.write('                       ')
			fh_c.write('filter[0][' + str(tall_filter_counter) + ']\n')
			tall_filter_counter += 1

	fh_c.write('\n')
	fh_c.write('    for(i = 0; i < ' + str(tall_filter_counter) + '; i++)\n')
	fh_c.write('      {\n')
	fh_c.write('        len = sprintf(buffer, "%s/Conv2D-%d.weights", argv[1], i);\n')
	fh_c.write('        buffer[len] = \'\\0\';\n')
	fh_c.write('        readWeights(buffer, &w);\n')
	fh_c.write('        setW_i_Conv2D(w, i, nn->convlayers);                        //  Set weights for filter[0][i]\n')
	fh_c.write('        free(w);\n')
	fh_c.write('      }\n')
	fh_c.write('\n')
	fh_c.write('    add_Conv2D(10, 5, nn);                                          //  Add a Conv2D layer that receives the WIDE input: 10 x 5\n')
	fh_c.write('    setName_Conv2D("Conv2D-Wide", nn->convlayers + 1);              //  Name the Conv2D layer\n')
	fh_c.write('\n')

	#################################################################  The Wide barrage
																	#  Then, sort Wide by height, then by width
	wide_items = sorted([x for x in conv_map.items() if x[0][0] == 'Wide'], key=lambda x: (x[0][0], x[0][1][1]))

	wide_filter_counter = 0
	wide_barrage_output_length = 0

	for k, v in wide_items:											#  Select operations by height
		quantity = v
		filter_w = k[1][0]
		filter_h = k[1][1]

		fh_py.write('\tconvWide'+str(filter_w)+'x'+str(filter_h)+' = Conv2D('+str(quantity)+', ('+str(filter_h)+', '+str(filter_w)+'), activation=\'relu\', padding=\'valid\', input_shape=(5, 10, 1), name=\'convWide'+str(filter_w)+'x'+str(filter_h)+'\')(wideEncode)\n')
		fh_py.write('\tflWide'+str(filter_w)+'x'+str(filter_h)+' = Flatten(name=\'flWide'+str(filter_w)+'x'+str(filter_h)+'\')(convWide'+str(filter_w)+'x'+str(filter_h)+')\n')

		if quantity > 1:											#  Lambda layers are for interleaving; don't bother if we only have one kernel per shape.
			for i in range(0, quantity):
				fh_py.write('\tlambdaWide'+str(filter_w)+'x'+str(filter_h)+'_'+str(i)+' = Lambda(lambda x: x[:, '+str(i)+'::'+str(quantity)+'], name=\'lambdaWide'+str(filter_w)+'x'+str(filter_h)+'_'+str(i)+'\')(flWide'+str(filter_w)+'x'+str(filter_h)+')\n')

		wide_barrage_output_length += output_shape('Wide', filter_w, filter_h, quantity)

		for q in range(0, quantity):
			fh_c.write('    add_Conv2DFilter(' + str(filter_w) + ', ' + str(filter_h) + ', nn->convlayers + 1);                     //  ')
			if q == 0:
				fh_c.write('Add ' + str(quantity) + ' (' + str(filter_w) + ' x ' + str(filter_h) + ') kernels: ')
			else:
				fh_c.write('                       ')
			fh_c.write('filter[1][' + str(wide_filter_counter) + ']\n')
			wide_filter_counter += 1

	fh_c.write('\n')
	fh_c.write('    for(i = 0; i < ' + str(wide_filter_counter) + '; i++)\n')
	fh_c.write('      {\n')
	fh_c.write('        len = sprintf(buffer, "%s/Conv2D-%d.weights", argv[1], i + ' + str(tall_filter_counter) + ');\n')
	fh_c.write('        buffer[len] = \'\\0\';\n')
	fh_c.write('        readWeights(buffer, &w);\n')
	fh_c.write('        setW_i_Conv2D(w, i, nn->convlayers + 1);                    //  Set weights for filter[1][i]\n')
	fh_c.write('        free(w);\n')
	fh_c.write('      }\n')
	fh_c.write('\n')

	#################################################################  Assemble into A SINGLE Concatenation
	fh_py.write('\tconvConcat = concatenate( [')

	for k, v in tall_items:											#  k = (Barrage, (w, h)); v = number of filters
		quantity = v
		for ctr in range(0, v):
			if quantity > 1:										#  Lambda layers are for interleaving; don't bother if we only have one kernel per shape.
				fh_py.write('lambdaTall'+str(k[1][0])+'x'+str(k[1][1])+'_'+str(ctr)+', ')
			else:													#  Else concatenate the outputs of the Flatten layers.
				fh_py.write('flTall'+str(k[1][0])+'x'+str(k[1][1])+', ')

	for k, v in wide_items:											#  k = (Barrage, (w, h)); v = number of filters
		quantity = v
		for ctr in range(0, v):
			if quantity > 1:										#  Lambda layers are for interleaving; don't bother if we only have one kernel per shape.
				fh_py.write('lambdaWide'+str(k[1][0])+'x'+str(k[1][1])+'_'+str(ctr)+', ')
			else:													#  Else concatenate the outputs of the Flatten layers.
				fh_py.write('flWide'+str(k[1][0])+'x'+str(k[1][1])+', ')

	fh_py.write('moveEncode] )\n')

	fh_c.write('    /******************************************************************************/\n')
	fh_c.write('    /***************************************************************    A C C U M */\n')
	fh_c.write('    add_Accum(' + str(tall_barrage_output_length + wide_barrage_output_length + 1) + ', nn);                                             //  Add accumulator layer (ACCUM_ARRAY, 0): receives Tall Conv Barrage + Wide Conv Barrage + to-move-indicator\n')
	fh_c.write('    setName_Accum("Accum-Tall-Wide", nn->accumlayers);              //  Name the accumulator layer\n')
	fh_c.write('\n')
	fh_c.write('    /******************************************************************************/\n')
	fh_c.write('    /***************************************************************    D E N S E */\n')

	#################################################################  Dense
	denseMap = [256, 64, 16]										#  NOT to include the final, single output!

	ctr = 0
	for denseLayerSpec in denseMap:
		if ctr == 0:
			fh_c.write('\n')
			fh_c.write('    add_Dense(' + str(tall_barrage_output_length + wide_barrage_output_length + 1) + ', ' + str(denseLayerSpec) + ', nn);                                        //  Add dense layer (DENSE_ARRAY, ' + str(ctr) + ')\n')
			fh_c.write('    setName_Dense("Dense-' + str(ctr) + '", nn->denselayers);                      //  Name the ' + str(ctr) + '-th dense layer\n')
			fh_c.write('    len = sprintf(buffer, "%s/Dense-' + str(ctr) + '.weights", argv[1]);\n')
			fh_c.write('    buffer[len] = \'\\0\';\n')
			fh_c.write('    readWeights(buffer, &w);\n')
			fh_c.write('    setW_Dense(w, nn->denselayers);\n')
			fh_c.write('    free(w);\n')

			fh_py.write('\tdense'+str(denseLayerSpec)+' = Dense('+str(denseLayerSpec)+', activation=\'relu\', name=\'dense'+str(denseLayerSpec)+'\')(convConcat)\n')

		else:
			fh_c.write('\n')
			fh_c.write('    add_Dense(' + str(denseMap[ctr - 1]) + ', ' + str(denseLayerSpec) + ', nn);                                        //  Add dense layer (DENSE_ARRAY, ' + str(ctr) + ')\n')
			fh_c.write('    setName_Dense("Dense-' + str(ctr) + '", nn->denselayers + ' + str(ctr) + ');                  //  Name the ' + str(ctr) + '-th dense layer\n')
			fh_c.write('    len = sprintf(buffer, "%s/Dense-' + str(ctr) + '.weights", argv[1]);\n')
			fh_c.write('    buffer[len] = \'\\0\';\n')
			fh_c.write('    readWeights(buffer, &w);\n')
			fh_c.write('    setW_Dense(w, nn->denselayers + ' + str(ctr) + ');\n')
			fh_c.write('    free(w);\n')

			fh_py.write('\tdense'+str(denseLayerSpec)+' = Dense('+str(denseLayerSpec)+', activation=\'relu\', name=\'dense'+str(denseLayerSpec)+'\')(dense'+str(denseMap[ctr - 1])+')\n')

		ctr += 1
																	#  The final output unit.
	fh_py.write('\tdense1 = Dense(1, activation=\'tanh\', name=\'dense1\')(dense'+str(denseMap[-1])+')\n')

	fh_c.write('\n')
	fh_c.write('    add_Dense(' + str(denseMap[-1]) + ', 1, nn);                                        //  Add final dense layer (DENSE_ARRAY, ' + str(len(denseMap)) + ')\n')
	fh_c.write('    setName_Dense("Dense-' + str(len(denseMap)) + '", nn->denselayers + ' + str(len(denseMap)) + ');                  //  Name the last dense layer\n')
	fh_c.write('    setF_i_Dense(HYPERBOLIC_TANGENT, 0, nn->denselayers + ' + str(len(denseMap)) + ');       //  Set output layer\'s activation function to hyperbolic tangent\n')
	fh_c.write('    len = sprintf(buffer, "%s/Dense-' + str(len(denseMap)) + '.weights", argv[1]);\n')
	fh_c.write('    buffer[len] = \'\\0\';\n')
	fh_c.write('    readWeights(buffer, &w);\n')
	fh_c.write('    setW_Dense(w, nn->denselayers + ' + str(len(denseMap)) + ');\n')
	fh_c.write('    free(w);\n')
	fh_c.write('\n')
	fh_c.write('    /******************************************************************************/\n')
	fh_c.write('\n')
	fh_c.write('    if(!linkLayers(INPUT_ARRAY, 0, 0, 50, CONV2D_ARRAY, 0, nn))     //  Connect input to conv2d[0], the Tall encode\n')
	fh_c.write('      printf(">>>                Link[0] failed\\n");\n')
	fh_c.write('    if(!linkLayers(INPUT_ARRAY, 0, 50, 100, CONV2D_ARRAY, 1, nn))   //  Connect input to conv2d[1], the Wide encode\n')
	fh_c.write('      printf(">>>                Link[1] failed\\n");\n')
	fh_c.write('\n')
	fh_c.write('    if(!linkLayers(CONV2D_ARRAY, 0, 0, ' + str(tall_barrage_output_length) + ', ACCUM_ARRAY, 0, nn))    //  Connect conv2d[0], the Tall encode, to the accumulator, accum[0]\n')
	fh_c.write('      printf(">>>                Link[2] failed\\n");\n')
	fh_c.write('    if(!linkLayers(CONV2D_ARRAY, 1, 0, ' + str(wide_barrage_output_length) + ', ACCUM_ARRAY, 0, nn))    //  Connect conv2d[1], the Wide encode, to the accumulator, accum[0]\n')
	fh_c.write('      printf(">>>                Link[3] failed\\n");\n')
	fh_c.write('\n')
	fh_c.write('    if(!linkLayers(INPUT_ARRAY, 0, 100, 101, ACCUM_ARRAY, 0, nn))   //  Connect input to the accumulator, accum[0]\n')
	fh_c.write('      printf(">>>                Link[4] failed\\n");\n')
	fh_c.write('\n')
	fh_c.write('    if(!linkLayers(ACCUM_ARRAY, 0, 0, ' + str(tall_barrage_output_length + wide_barrage_output_length + 1) + ', DENSE_ARRAY, 0, nn))     //  Connect accum[0], the accumulator, to dense[0], Dense-' + str(denseMap[0]) + '\n')
	fh_c.write('      printf(">>>                Link[5] failed\\n");\n')
	fh_c.write('\n')
	fh_c.write('    if(!linkLayers(DENSE_ARRAY, 0, 0, ' + str(denseMap[0]) + ', DENSE_ARRAY, 1, nn))     //  Connect dense[0], Dense-' + str(denseMap[0]) + ', to dense[1], Dense-' + str(denseMap[1]) + '\n')
	fh_c.write('      printf(">>>                Link[6] failed\\n");\n')
	fh_c.write('\n')
	fh_c.write('    if(!linkLayers(DENSE_ARRAY, 1, 0, ' + str(denseMap[1]) + ', DENSE_ARRAY, 2, nn))     //  Connect dense[1], Dense-' + str(denseMap[1]) + ', to dense[2], Dense-' + str(denseMap[2]) + '\n')
	fh_c.write('      printf(">>>                Link[7] failed\\n");\n')
	fh_c.write('\n')
	fh_c.write('    if(!linkLayers(DENSE_ARRAY, 2, 0, ' + str(denseMap[2]) + ', DENSE_ARRAY, 3, nn))     //  Connect dense[2], Dense-' + str(denseMap[2]) + ', to dense[3], Dense-1\n')
	fh_c.write('      printf(">>>                Link[8] failed\\n");\n')
	fh_c.write('\n')
	fh_c.write('    sortEdges(nn);\n')
	fh_c.write('    printEdgeList(nn);\n')
	fh_c.write('    printf("\\n\\n");\n')
	fh_c.write('    len = sprintf(buffer, "Lodz Kaliska, trained using TD lambda");\n')
	fh_c.write('    i = 0;\n')
	fh_c.write('    while(i < len && i < COMMSTR_LEN)\n')
	fh_c.write('      {\n')
	fh_c.write('        nn->comment[i] = buffer[i];\n')
	fh_c.write('        i++;\n')
	fh_c.write('      }\n')
	fh_c.write('    while(i < COMMSTR_LEN)\n')
	fh_c.write('      {\n')
	fh_c.write('        nn->comment[i] = \'\\0\';\n')
	fh_c.write('        i++;\n')
	fh_c.write('      }\n')
	fh_c.write('    nn->gen = (unsigned int)atoi(argv[1]);\n')
	fh_c.write('    //nn->fit = (double)atof(argv[2]);                                                   //  Set this yourself.\n')
	fh_c.write('    print_NN(nn);\n')
	fh_c.write('\n')
	fh_c.write('    len = sprintf(buffer, "lodzkaliska-%s.nn", argv[1]);\n')
	fh_c.write('    buffer[len] = \'\\0\';\n')
	fh_c.write('\n')
	fh_c.write('    write_NN(buffer, nn);\n')
	fh_c.write('    free_NN(nn);\n')
	fh_c.write('\n')
	fh_c.write('    return 0;\n')
	fh_c.write('  }\n')
	fh_c.write('\n')
	fh_c.write('/* Open the given file, read its weights into the given \'buffer,\' and return the length of \'buffer.\' */\n')
	fh_c.write('unsigned int readWeights(char* filename, double** buffer)\n')
	fh_c.write('  {\n')
	fh_c.write('    FILE* fp;\n')
	fh_c.write('    unsigned int len = 0;\n')
	fh_c.write('    double x;\n')
	fh_c.write('\n')
	fh_c.write('    printf("Reading %s:", filename);\n')
	fh_c.write('\n')
	fh_c.write('    if((fp = fopen(filename, "rb")) == NULL)\n')
	fh_c.write('      {\n')
	fh_c.write('        printf("ERROR: Unable to open file\\n");\n')
	fh_c.write('        exit(1);\n')
	fh_c.write('      }\n')
	fh_c.write('    fseek(fp, 0, SEEK_SET);                                         //  Rewind\n')
	fh_c.write('    while(!feof(fp))\n')
	fh_c.write('      {\n')
	fh_c.write('        if(fread(&x, sizeof(double), 1, fp) == 1)\n')
	fh_c.write('          {\n')
	fh_c.write('            if(++len == 1)\n')
	fh_c.write('              {\n')
	fh_c.write('                if(((*buffer) = (double*)malloc(sizeof(double))) == NULL)\n')
	fh_c.write('                  {\n')
	fh_c.write('                    printf("ERROR: Unable to malloc buffer\\n");\n')
	fh_c.write('                    exit(1);\n')
	fh_c.write('                  }\n')
	fh_c.write('              }\n')
	fh_c.write('            else\n')
	fh_c.write('              {\n')
	fh_c.write('                if(((*buffer) = (double*)realloc((*buffer), len * sizeof(double))) == NULL)\n')
	fh_c.write('                  {\n')
	fh_c.write('                    printf("ERROR: Unable to realloc buffer\\n");\n')
	fh_c.write('                    exit(1);\n')
	fh_c.write('                  }\n')
	fh_c.write('              }\n')
	fh_c.write('            (*buffer)[len - 1] = x;\n')
	fh_c.write('          }\n')
	fh_c.write('      }\n')
	fh_c.write('    printf(" %d weights\\n", len);\n')
	fh_c.write('    fclose(fp);                                                     //  Close the file\n')
	fh_c.write('\n')
	fh_c.write('    return len;\n')
	fh_c.write('  }\n')
	fh_c.close()													#  Done building C code.

	fh_py.write('\tmodel = models.Model( [ tallEncode, wideEncode, moveEncode ], [ dense1 ] )\n')
	fh_py.write('\tmodel.compile(optimizer=keras.optimizers.Adagrad(learning_rate=' + str(params['learning-rate']) + '), loss=\'mse\', metrics=[\'acc\'])\n')
	fh_py.write('\treturn model\n')
	fh_py.close()													#  Done building Python code.

	return

#  (input_w - filter_w + 1) * (input_h - filter_h + 1) * filter_count
def output_shape(barrage, filter_w, filter_h, filter_count):
	if barrage == 'Tall':
		input_w = 5
		input_h = 10
	elif barrage == 'Wide':
		input_w = 10
		input_h = 5
	return (input_w - filter_w + 1) * (input_h - filter_h + 1) * filter_count

def get_command_line_params():
	params = {}

	params['learning-rate'] = 0.0001								#  Default.

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-lr', '-v', '-?', '-help', '--help']
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
				if argtarget == '-lr':
					params['learning-rate'] = float(argval)

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Generate model-building code in C and Python according to the model specs.')
	print('')
	print('Usage:  python3 build_model_code.py <parameters, preceded by flags>')
	print(' e.g.:  python3 build_model_code.py -lr 0.001 -v')
	print('')
	print('Flags:  -lr   Learning rate. Default is 0.0001.')
	print('')
	print('        -v    Enable verbosity')
	print('        -?    Display this message')

if __name__ == '__main__':
	main()
