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
    if len(sys.argv) != 4:
        print ('Error: invalid number of arguments in', sys.argv)
        print ('Usage: spectrogram.py -train GENRE PATH')
        print ('Genres: dnb | house | dstep')
        sys.exit()

    if sys.argv[1] == '-train':
        songs_list = os.listdir(sys.argv[3])
        print(songs_list)
        sg = Spectrogram(display=True)
        pp = preProcess()
        for f in songs_list:
            #extract the mel perc and chroma specs
            #take all numpy data and concatenate eg: {[mel], [perc], [chroma]}
            #compute each through nn
            print(f)
            spec_master = []
            mel_spec = sg.mel_spectrogram(mp3=os.path.join(sys.argv[3], f))
            #mel_slice = sg.process_np_data(mel_spec[1]) #process S data
            spec_master.append(mel_spec)
            #display_sg((spec[0],slice,spec[2],spec[3]))

            perc_spec = sg.perc_spectrogram(mp3=os.path.join(sys.argv[3], f))
            #perc_slice = sg.process_np_data(perc_spec[1]) #process logSp data
            spec_master.append(perc_spec)

            harm_spec = sg.harm_spectrogram(mp3=os.path.join(sys.argv[3], f))
            #harm_slice = sg.process_np_data(harm_spec[1]) #process logSh data
            spec_master.append(harm_spec)

            chroma_spec = sg.chromagram(mp3=os.path.join(sys.argv[3], f))
            #chroma_slice = sg.process_np_data(chroma_spec[1]) #process C data
            spec_master.append(chroma_spec)

            #print(spec_master)

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

if __name__ == '__main__':
    main()