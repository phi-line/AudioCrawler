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

import operator

from preprocess import preProcess

class Spectrogram:
    def __init__(self, display = False, trim = True, slice = False, offset =
    30,
                 duration=60):
        self.display = display
        self.trim = trim
        self.slice = slice
        self.offset = offset
        self.duration = duration

    def mel_spectrogram(self, mp3='test.mp3'):
        '''
        this function displays a mel spectrogram .csv of the mp3 data
        :return: ('mel', y, sr, S, log_S)
        :   'mel': str
        :   S    : ndarray
        :   y    : ndarray
        :   sr   : int
        '''
        if mp3[-4:] != '.mp3':
            print("could not load path:", mp3)
            return
        else:
            if self.trim:
                y, sr = librosa.load(path=mp3, offset=self.offset,
                                    duration=self.duration)
            else: y, sr = librosa.load(path=mp3)

            if self.slice:
                y = self.slice_onset(y)

        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

        if self.slice:
            pp = preProcess
            S = pp.z_norm(S)
        else: S = self.process_np_data(S)

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.logamplitude(S, ref_power=np.max)

        # display
        if self.display:
            # Make a new figure
            plt.figure(figsize=(12, 4))

            # Display the spectrogram on a mel scale
            # sample rate and hop length parameters are used to render the time axis
            librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

            # Put a descriptive title on the plot
            plt.title('mel power spectrogram ')

            # draw a color bar
            plt.colorbar(format='%+02.0f dB')

            # Make the figure layout compact
            plt.tight_layout()
            plt.show() #save to output
            # plt.savefig(mp3 + '.png')

        # generate tuple of S and log_S
        spec = ('mels', S, y, sr)
        return spec

    def perc_spectrogram(self, mp3='test.mp3', slice = True):
        if mp3[-4:] != '.mp3':
            print("could not load path:", mp3)
            return
        else:
            if self.trim:
                y, sr = librosa.load(path=mp3, offset=self.offset,
                                    duration=self.duration)
            else: y, sr = librosa.load(path=mp3)

            if self.slice:
                y = self.slice_onset(y)

        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # What do the spectrograms look like?
        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        #S_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr)
        S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

        if self.slice:
            pp = preProcess
            S_percussive = pp.z_norm(S_percussive)
        else:
            S_percussive = self.process_np_data(S_percussive)

        # Convert to log scale (dB). We'll use the peak power as reference.
        #log_Sh = librosa.logamplitude(S_harmonic, ref_power=np.max)
        log_Sp = librosa.logamplitude(S_percussive, ref_power=np.max)

        # display
        if self.display:
            # Make a new figure
            plt.figure(figsize=(12, 6))

            # plt.subplot(2, 1, 1)
            # # Display the spectrogram on a mel scale
            # #librosa.display.specshow(log_Sh, sr=sr, y_axis='mel')
            #
            # # Put a descriptive title on the plot
            # plt.title('mel power spectrogram (Harmonic)')
            #
            # # draw a color bar
            # plt.colorbar(format='%+02.0f dB')

            # plt.subplot(2, 1, 2)
            librosa.display.specshow(log_Sp, sr=sr, x_axis='time',
                                     y_axis='mel')

            # Put a descriptive title on the plot
            plt.title('mel power spectrogram (Percussive)')

            # draw a color bar
            plt.colorbar(format='%+02.0f dB')

            # Make the figure layout compact
            plt.tight_layout()
            plt.show()

        # generate tuple of S and log_S
        spec = ('percussion', log_Sp, y, sr)
        return spec

    def harm_spectrogram(self, mp3='test.mp3', slice = True):
        if mp3[-4:] != '.mp3':
            print("could not load path:", mp3)
            return
        else:
            if self.trim:
                y, sr = librosa.load(path=mp3, offset=self.offset,
                                    duration=self.duration)
            else: y, sr = librosa.load(path=mp3)

            if self.slice:
                y = self.slice_onset(y)

        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # What do the spectrograms look like?
        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr)
        #S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

        if self.slice:
            pp = preProcess
            S_harmonic = pp.z_norm(S_harmonic)
        else:
            S_harmonic = self.process_np_data(S_harmonic)

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_Sh = librosa.logamplitude(S_harmonic, ref_power=np.max)
        #log_Sp = librosa.logamplitude(S_percussive, ref_power=np.max)

        # display
        if self.display:
            # Make a new figure
            plt.figure(figsize=(12, 6))

            # plt.subplot(2, 1, 1)
            # Display the spectrogram on a mel scale
            librosa.display.specshow(log_Sh, sr=sr, y_axis='mel')

            # Put a descriptive title on the plot
            plt.title('mel power spectrogram (Harmonic)')

            # draw a color bar
            plt.colorbar(format='%+02.0f dB')

            # plt.subplot(2, 1, 2)
            # # librosa.display.specshow(log_Sp, sr=sr, x_axis='time',
            # #                          y_axis='mel')
            #
            # # Put a descriptive title on the plot
            # plt.title('mel power spectrogram (Percussive)')
            #
            # # draw a color bar
            # plt.colorbar(format='%+02.0f dB')

            # Make the figure layout compact
            plt.tight_layout()
            plt.show()

        # generate tuple of S and log_S
        spec = ('harmonic', log_Sh, y, sr)
        return spec

    #depreciated - not using chromagram
    def chromagram(self, mp3='test.mp3', slice = True):
        if mp3[-4:] != '.mp3':
            print("could not load path:", mp3)
            return
        else:
            y, sr = librosa.load(path=mp3)

        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # What do the spectrograms look like?
        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr)
        #S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

        if slice:
            S_harmonic = self.process_np_data(S_harmonic)

        # We'll use a CQT-based chromagram here.  An STFT-based implementation also exists in chroma_cqt()
        # We'll use the harmonic component to avoid pollution from transients
        C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

        # display
        if self.display:
            # Make a new figure
            plt.figure(figsize=(12, 4))

            # Display the chromagram: the energy in each chromatic pitch class as a function of time
            # To make sure that the colors span the full range of chroma values, set vmin and vmax
            librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma',
                                     vmin=0,
                                     vmax=1)

            plt.title('Chromagram')
            plt.colorbar()

            plt.tight_layout()
            plt.show()

        # generate tuple of S and log_S
        spec = ('chroma', C, y, sr)
        return spec

    #depreciated - not using beat diagram
    def beat_gram(self, mp3='test.mp3'):
        if mp3[-4:] != '.mp3':
            print("could not load path:", mp3)
            return
        else:
            y, sr = librosa.load(path=mp3)
        y_harmonic, y_percussive = librosa.effects.hpss(y)

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
        if self.display:
            plt.show()

    def process_y_data(self, y_arr):
        '''
        Takes numpy array as a parameter to apply filter to data
        Filters numpy data down to slice and then applies filter
        :param np_arr: unprocessed
        :return: np_array: processed
        '''
        PERCENT_REC = .4  # the percentage of the np array to capture from
        REC_SCALE = 32  # the number of X steps to record from start

        pp = preProcess()
        half = int(len(y_arr) * PERCENT_REC)
        slice = y_arr[half:(half + REC_SCALE)]
        return pp.z_norm(slice)

    def process_np_data(self, np_arr):
        '''
        Takes numpy array as a parameter to apply filter to data
        Filters numpy data down to slice and then applies filter
        :param np_arr: unprocessed
        :return: np_array: processed
        '''
        PERCENT_REC = .4  # the percentage of the np array to capture from
        REC_SCALE = 32  # the number of X steps to record from start

        pp = preProcess()
        #print(np_arr.shape)
        half = int(len(np_arr) * PERCENT_REC)
        slice = np_arr[:, half:(half + REC_SCALE)]
        #print(slice.shape)
        return pp.z_norm(slice)

    def slice_onset(self, y_arr, sr = 22050):
        '''
        this function takes in the y data, 
        '''
        # use pre-computed onset envelope
        o_env = librosa.onset.onset_strength(y_arr, sr=sr)
        size = o_env.shape[0]

        # take middle 20% and trim the rest
        start    = 0 #int((size*.3)/2)
        stop     = size #int(size-start)
        new_env  = o_env[start:stop]
        n_chunks = int(len(new_env)/32)
        if n_chunks%2!=0: # ensure even parity
            n_chunks+=1

        # split into chunks of 32
        chunks = []; counter = 0
        for c in range(n_chunks):
            chunks.append(new_env[counter:counter+32])
            counter += 32

        # compute average for each chunk
        chunk_means = []
        for chunk in chunks:
            chunk_means.append(np.mean(chunk))

        # get differences between pairs
        differences = []
        for c in range(int(len(chunk_means)/2)):
            differences.append(chunk_means[2*c]-chunk_means[2*c+1])

        # get max difference and corresponding chunks
        max_diff = np.max(differences)
        diff_index, diff_val = max(enumerate(chunk_means), key=operator.itemgetter(1))
        chunks_oi = (diff_index*2, diff_index*2+1)

        # get interval percentages
        start_percent = (start+chunks_oi[0]*32)/size
        stop_percent  = 1-(size-(start+chunks_oi[1]*32))/size

        y_length = y_arr.shape[0]
        y_start  = int(y_length*start_percent)
        y_stop   = int(y_length*stop_percent)

        # get slice and return
        slice = y_arr[y_start:y_stop]
        return slice


#old main code
def main():
    if len(sys.argv) < 3:
        print ('Error: invalid number of arguments')
        print ('Usage: spectrogram.py TYPE PATH')
        print ('Types: mel | perc | chroma | beat')
        sys.exit()
    # check for single song case
    elif sys.argv[2][-4:] == '.mp3':
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
    #batch directory
    else:
        songs_list = os.listdir(sys.argv[2])
        sg = Spectrogram()
        for f in songs_list:
            print(f)
            spec_master = []
            if sys.argv[1] == 'mel':
                spec = sg.mel_spectrogram(mp3 = os.path.join(sys.argv[2], f))
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

if __name__ == '__main__':
    main()