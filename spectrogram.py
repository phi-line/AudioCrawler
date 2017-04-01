'''
This program takes audio spectrogram data from a song or a whole folder of
songs and outputs to a numpy array.

Usage: spectrogram.py TYPE PATH
Types: mel | perc | chroma
'''
from __future__ import print_function

import sys #kwargs
import os 

import numpy as np 
import scipy as sp 
# import keras only if we need DNN
import matplotlib.pyplot as plt

# Librosa is a music analysis tool
# https://github.com/librosa/librosa
import librosa
import librosa.display

def main():
    if len(sys.argv) < 3:
        print ('Error: invalid number of arguments')
        print ('Usage: spectrogram.py TYPE PATH')
        print ('Types: mel | perc | chroma | beat')
        sys.exit()
    # check for single song case
    elif sys.argv[2][-4:] == '.mp3':
        print(sys.argv[2])
        if sys.argv[1] == 'mel':
            mel_spectrogram(mp3=sys.argv[2], display=True)
        elif sys.argv[1] == 'perc':
            perc_spectrogram(mp3=sys.argv[2], display=True)
        elif sys.argv[1] == 'chroma':
            chromagram(mp3=sys.argv[2], display=True)
        elif sys.argv[1] == 'beat':
            beat_gram(mp3=sys.argv[2], display=True)
        else:
            print('Invalid type given:', sys.argv[1])
            print('Types: mel perc chroma')
    #batch directory
    else:
        songs_list = os.listdir(sys.argv[2])
        for f in songs_list:
            print(f)
            if sys.argv[1] == 'mel':
                mel_spectrogram(mp3 = os.path.join(sys.argv[2], f), display=True)
            elif sys.argv[1] == 'perc':
                perc_spectrogram(mp3 = os.path.join(sys.argv[2], f), display=True)
            elif sys.argv[1] == 'chroma':
                chromagram(mp3 = os.path.join(sys.argv[2], f), display=True)
            elif sys.argv[1] == 'beat':
                beat_gram(mp3=sys.argv[2], display=True)
            else:
                print('Invalid type given:', sys.argv[1])
                print('Types: mel | perc | chroma | beat')


def mel_spectrogram(mp3 = sys.argv[2], display = True):
    '''
    this function displays a mel spectrogram .csv of the mp3 data
    :return:
    '''
    y, sr = librosa.load(path=mp3)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.logamplitude(S, ref_power=np.max)

    # Make a new figure
    plt.figure(figsize=(12,4))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    # Put a descriptive title on the plot
    plt.title('mel power spectrogram')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()

    # display
    if display:
        plt.show()

def perc_spectrogram(mp3 = sys.argv[2], display = True):
    y, sr = librosa.load(path=mp3)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # What do the spectrograms look like?
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr)
    S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_Sh = librosa.logamplitude(S_harmonic, ref_power=np.max)
    log_Sp = librosa.logamplitude(S_percussive, ref_power=np.max)

    # Make a new figure
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    # Display the spectrogram on a mel scale
    librosa.display.specshow(log_Sh, sr=sr, y_axis='mel')

    # Put a descriptive title on the plot
    plt.title('mel power spectrogram (Harmonic)')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(log_Sp, sr=sr, x_axis='time', y_axis='mel')

    # Put a descriptive title on the plot
    plt.title('mel power spectrogram (Percussive)')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()

    # display
    if display:
        plt.show()

def chromagram(mp3 = sys.argv[2], display = True):
    y, sr = librosa.load(path=mp3)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # What do the spectrograms look like?
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr)
    S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

    # We'll use a CQT-based chromagram here.  An STFT-based implementation also exists in chroma_cqt()
    # We'll use the harmonic component to avoid pollution from transients
    C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

    # Make a new figure
    plt.figure(figsize=(12, 4))

    # Display the chromagram: the energy in each chromatic pitch class as a function of time
    # To make sure that the colors span the full range of chroma values, set vmin and vmax
    librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0,
                             vmax=1)

    plt.title('Chromagram')
    plt.colorbar()

    plt.tight_layout()

    # display
    if display:
        plt.show()

def beat_gram(mp3 = sys.argv[2], display = True):
    y, sr = librosa.load(path=mp3)
    y_percussive = librosa.effects.hpss(y)

    # What do the spectrograms look like?
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_Sp = librosa.logamplitude(S_percussive, ref_power=np.max)

    # Now, let's run the beat tracker.
    # We'll use the percussive component for this part
    plt.figure(figsize=(12, 6))
    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # Let's re-draw the spectrogram, but this time, overlay the detected beats
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_Sp, sr=sr, x_axis='time', y_axis='mel')

    # Let's draw transparent lines over the beat frames
    plt.vlines(librosa.frames_to_time(beats),
               1, 0.5 * sr,
               colors='w', linestyles='-', linewidth=2, alpha=0.1)

    plt.axis('tight')

    plt.colorbar(format='%+02.0f dB')

    plt.tight_layout()

    # display
    if display:
        plt.show()


if __name__ == '__main__':
    main()