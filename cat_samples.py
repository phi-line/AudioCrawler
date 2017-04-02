
master_data = [] # master list of data

def cat_samples(master_data):
	'''
	:param: master_data - list of tuples
			each tuple contains 2 elements: tuple, genre
				tuple: (mel, perc, harm)
				genre: str
	'''
	# Initialize data lists
	mel   = []
	perc  = []
	harm  = []
	genre = []
	n = len(data[0][0])

	# Decompose the data into genre and spec data, for list members (!genre) the dtype is ndarray
	for data in master_data:
		mel.append(data[0][0])
		perc.append(data[0][1])
		harm.append(data[0][2])
		genre.append(data[1])

	# Now we have lists of either 1D arrays (mel, perc, harm) or str (genre)
	# Next we convert each list to a ndarray
	mel   = vec2matrix(mel,  ncol=n)
	perc  = vec2matrix(perc, ncol=n)
	harm  = vec2matrix(harm, ncol=n)
	genre = vec2matrix(genre,ncol=n)

	data = ((mel, perc, harm), genre)
	return data

   




