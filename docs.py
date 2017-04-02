'''
keras tutorial:

# load_data returns 2 tuples: 
	x_train, x_test - list of sequences, which are lists of ints/indexes
	y_train, y_test - list of int labels (0 or 1)
# sequence returns 2d np array from arguments inside tuple
	takes in sequences argument - a list of lists of ints
# fit takes in two 2d np

fit(x_train, y_train, validation = (x_test, y_test))

--------------------------------------------------


intended: pass in EACH SAMPLE

then append each sample into one big sample??

sample data: one input mp3 file -> tuple(string, tuple(mel, perc, harm, chroma))
	such that mel, perc, harm, and chroma are 2d arrays
	accessible through data[][]:
		ex. data[1][0] accesses mel.
	x_train = string
	y_train = tuple of input files (2d array)
	(x_train, y_train)

sequence converts a tuple into a 2d array
	=> ONLY take in y_train and y_test

or... we append the 2d arrays into one np array to pass on as y.
ex: master_mel = []
	master_mel.append(temp_mel)


fit - trains model for # of epochs
	validation_data takes in tuple, such as (x_test, y_test)
		x: numpy array or list of numpy arrays (if model has multiple inputs)
		y: numpy array

therefore, regardless of input: we HAVE to train on two 2d arrays.

trim piece of song -> average towards 1d array
rows now samples
columns treated as features (time)
'''