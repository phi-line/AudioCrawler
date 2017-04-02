'''
This is the main driver for our spectrogram and the neural net
Also, cats have great hearing hence curiouscat.py
'''

from __future__ import print_function

import os
import sys #args
from random import seed, shuffle

import numpy as np
import scipy as sp
# import keras only if we need DNN
import matplotlib.pyplot as plt

# Librosa is a music analysis tool
# https://github.com/librosa/librosa
import librosa
import librosa.display

from glob import glob

from spectrogram import Spectrogram
from preprocess import preProcess

def main():
    if len(sys.argv) != 3:
        print ('Error: invalid number of arguments in', sys.argv)
        print ('Usage: spectrogram.py -train PATH')
        sys.exit()

    seed()
    if sys.argv[1] == '-train':
        #needs to recursively add in sound data
        # songs = {os.path.join(r, f):s for r, s, files in os.walk(
        #     sys.argv[2]) for f in files if os.path.splitext(f)[1] == '.mp3'}

        songs = []
        walk_dir = os.path.abspath(sys.argv[2])
        n_genres = 0
        for root, subs, files in os.walk(sys.argv[2]): #rootdir
            for sub in subs:
                n_genres += 1
            for f in files:
                if os.path.splitext(f)[1] == '.mp3':
                    songs.append((os.path.join(root, f), root[6:]))
                    path = os.path.normpath(os.path.join(root, f))
                    path.split(os.sep)

        #songs = os.listdir(sys.argv[3])
        print ('Loaded {} songs and {} genres'.format(len(songs),n_genres))
        shuffle(songs)
        sg = Spectrogram(display=False)

        master_data = []  # master list of data

        for f in songs:
            genre = f[1]
            #extract the mel perc and chroma specs
            #take all numpy data and concatenate eg: {[mel], [perc], [chroma]}
            #compute each through nn
            update = update_info(f[0], 3)

            #mel
            print(update.next(), end='\r')
            spec_master = []
            mel_spec = sg.mel_spectrogram(mp3=f[0])
            spec_master.append(mel_spec)

            #spec
            print(update.next(), end='\r')
            perc_spec = sg.perc_spectrogram(mp3=f[0])
            spec_master.append(perc_spec)

            #harm
            print(update.next(), end='\r')
            harm_spec = sg.harm_spectrogram(mp3=f[0])
            spec_master.append(harm_spec)

            #chroma
            # print(update.next(), end='\r')
            # chroma_spec = sg.chromagram(mp3=os.path.join(sys.argv[2], f))
            # spec_master.append(chroma_spec)
            # print("\n\r", end="")

            n = mel_spec[1].shape[1]
            data_tuple = (tuple(spec_master), genre, n)
            master_data.append(data_tuple)
            #print(data_tuple)

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
	mel   = np.reshape(mel,  ncol=n) #vec2matrix
	perc  = np.reshape(perc, ncol=n)
	harm  = np.reshape(harm, ncol=n)
	genre = np.reshape(genre,ncol=n)

	data = ((mel, perc, harm), genre)
	return data

def display_sg(data_tuple):
    plt.figure(figsize=(12, 4))
    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(data_tuple[1], sr=data_tuple[3], x_axis='time',
                             y_axis='mel')

    # Put a descriptive title on the plot
    plt.title('spectrogram')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')
    # Make the figure layout compact
    plt.tight_layout()

    plt.show()

class update_info(object):
    def __init__(self, song, n):
        self.song = song
        self.n = n
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n:
            self.num += 1
            cur = '{0} [{1}/{2}]'.format(self.song, self.num, self.n)
            return cur
        else:
            raise StopIteration()

if __name__ == '__main__':
    main()