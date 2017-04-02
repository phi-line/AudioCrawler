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

import rAnalyser

from glob import glob

from spectrogram import Spectrogram
from preprocess import preProcess

def main():
    if len(sys.argv) != 4:
        print ('Error: invalid number of arguments in', sys.argv)
        print ('Usage: spectrogram.py -train PATH INTERVAL')
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

        n = sys.argv[3]
        master_data = []  # master list of data

        i = 1
        for f in songs:
            genre = f[1]
            #extract the mel perc and chroma specs
            #take all numpy data and concatenate eg: {[mel], [perc], [chroma]}
            #compute each through nn
            update = update_info(f[0], i, 3)

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
            print("\n\r", end="")
            i += 1

            #n = mel_spec[1].shape[1]

            #((mel, perc, harm), genre, n)
            data_tuple = (tuple(spec_master), genre, n)
            master_data.append(data_tuple)
            #print(data_tuple)

        jsonify(master_data)

        ai = rAnalyser.smRegAlog(n)
        for data in master_data:
            ai.teachAI(data)

def jsonify(master_data):
    import codecs, json, time

    json_list = []
    for data_tuple in master_data:
        spec_list = []
        for spec in data_tuple[0]:
            print(type(spec))
            np_to_list = spec.tolist()
            spec_list.append(np_to_list)
        json_list.extend( [spec_list, data_tuple[1], data_tuple[2]] )

    file_path = "dumps/dump_{}.json".format(time.strftime("%Y%m%d-%H%M%S"))
    print("dumped to", file_path)
    ## your path variable
    json.dump(json_list, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

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
    def __init__(self, song, i, step):
        self.song = song
        self.n = step
        self.i = i
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n:
            self.num += 1
            cur = ' {} | {} [{}/{}]'.format(self.i, self.song, self.num,
                                            self.n)
            return cur
        else:
            raise StopIteration()

if __name__ == '__main__':
    main()