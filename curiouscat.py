'''
This is the main driver for our spectrogram and the neural net
Also, cats have great hearing hence curiouscat.py
'''

from __future__ import print_function

import sys #args
import os

import numpy as np
import scipy as sp
# import keras only if we need DNN
import matplotlib.pyplot as plt

# Librosa is a music analysis tool
# https://github.com/librosa/librosa
import librosa
import librosa.display

from spectrogram import Spectrogram
from preprocess import preProcess

def main():
    if len(sys.argv) < 3:
        print ('Error: invalid number of arguments')
        print ('Usage: spectrogram.py TYPE PATH')
        print ('Types: mel | perc | chroma | beat')
        sys.exit()


    if sys.argv[1] == '-train':
        songs_list = os.listdir(sys.argv[2])
        sg = Spectrogram()
        for f in songs_list:
            print(f)
            spec_master = []
            if sys.argv[1] == 'mel':
                spec = sg.mel_spectrogram(mp3=os.path.join(sys.argv[2], f))
                slice = process_np_data(spec)
                spec_master.append(slice)
            elif sys.argv[1] == 'perc':
                spec = sg.perc_spectrogram(mp3=os.path.join(sys.argv[2], f))
                slice = process_np_data(spec)
                spec_master.append(slice)
            elif sys.argv[1] == 'chroma':
                spec = sg.chromagram(mp3=os.path.join(sys.argv[2], f))
                slice = process_np_data(spec)
                spec_master.append(slice)
            elif sys.argv[1] == 'beat':
                spec = sg.beat_gram(mp3=os.path.join(sys.argv[2], f))
                slice = process_np_data(spec)
                spec_master.append(slice)
            else:
                print('Invalid type given:', sys.argv[1])
                print('Types: mel | perc | chroma | beat')
        print(spec_master)


PERCENT_REC = 4 #the percentage of the np array to capture from
REC_SCALE = 32 # the number of X steps to record from start

def process_np_data(np_arr):
    '''
    Takes numpy array as a parameter to apply filter to data
    Filters numpy data down to slice and then applies filter
    :param np_arr: unprocessed
    :return: np_array: processed
    '''
    pp = preProcess()
    half = int(len(np_arr)/PERCENT_REC)
    slice = np_arr[half:half+REC_SCALE]
    return pp.z_norm(slice)

if __name__ == '__main__':
    main()