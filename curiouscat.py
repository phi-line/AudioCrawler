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

def main():
    if len(sys.argv) < 3:
        print ('Error: invalid number of arguments')
        print ('Usage: spectrogram.py TYPE PATH')
        print ('Types: mel | perc | chroma | beat')
        sys.exit()
    #batch directory
    elif sys.argv[2][-4:] != '.mp3':
        songs_list = os.listdir(sys.argv[2])
        sg = Spectrogram()
        for f in songs_list:
            print(f)
            spec_master = []
            if sys.argv[1] == 'mel':
                spec = sg.mel_spectrogram(mp3 = os.path.join(sys.argv[2], f))
                spec = sg.z_norm(spec)
                spec_master.append(spec)
            elif sys.argv[1] == 'perc':
                spec = sg.perc_spectrogram(mp3 = os.path.join(sys.argv[2], f))
                spec_master.append(spec)
            elif sys.argv[1] == 'chroma':
                spec = sg.chromagram(mp3 = os.path.join(sys.argv[2], f))
                spec_master.append(spec)
            elif sys.argv[1] == 'beat':
                spec = sg.beat_gram(mp3 = os.path.join(sys.argv[2], f))
                spec_master.append(spec)
            else:
                print('Invalid type given:', sys.argv[1])
                print('Types: mel | perc | chroma | beat')
        print(spec_master)
    # check for single song case
    else:
        print(sys.argv[2])
        sg = Spectrogram(display=True)
        if sys.argv[1] == 'mel':
            sg.mel_spectrogram(mp3=sys.argv[2])
        elif sys.argv[1] == 'perc':
            sg.perc_spectrogram(mp3=sys.argv[2])
        elif sys.argv[1] == 'chroma':
            sg.chromagram(mp3=sys.argv[2])
        elif sys.argv[1] == 'beat':
            sg.beat_gram(mp3=sys.argv[2])
        else:
            print('Invalid type given:', sys.argv[1])
            print('Types: mel | perc | chroma | beat')
            sys.exit()

if __name__ == '__main__':
    main()