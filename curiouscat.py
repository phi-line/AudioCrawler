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
import rAnalyserLargeTuple

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
                    print(root[7:])
                    songs.append((os.path.join(root, f), root[7:]))
                    path = os.path.normpath(os.path.join(root, f))
                    path.split(os.sep)

        #songs = os.listdir(sys.argv[3])
        print ('Loaded {} songs and {} genres'.format(len(songs),n_genres))
        #shuffle(songs)
        sg = Spectrogram(display=True, trim=True, slice=False,
                         offset=50, duration=60)

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
            #print("\n\r", end="")
            sys.stdout.write('\x1b[2K\r')
            i += 1

            #n = mel_spec[1].shape[1]

            #((mel, perc, harm), genre, n)
            data_tuple = (tuple(spec_master), genre, n)

            #nm_model = train_model(data_tuple[0][0][1], data_tuple[1])
            master_data.append(data_tuple)
            #print(data_tuple)


        #jsonify(master_data)

        #all_data = cat_samples(master_data)
        #ai = rAnalyserLargeTuple.smRegAlog(all_data)


        #ai = rAnalyser.smRegAlog() #n
        # for data in master_data:
        #     ai.teachAI(data)

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

    # tuple -> mel -> numpy arr
    #print(master_data[0][0][0][1],type(master_data[0][0][0][1]),
          #sep='\n\n\n')
    n = master_data[0][0][0][1].shape
    print(n)

    # Decompose the data into genre and spec data, for list members (!genre) the dtype is ndarray
    for data in master_data:
        mel.append(data[0][0][1])
        perc.append(data[0][1][1])
        harm.append(data[0][2][1])
        genre.append(data[1])

    # Now we have lists of either 1D arrays (mel, perc, harm) or str (genre)
    # Next we convert each list to a ndarray
    print(np.asarray(mel))
    mel   = np.reshape(np.asarray(mel),  n) #ncol
    perc  = np.reshape(np.asarray(perc), n)
    harm  = np.reshape(np.asarray(harm), n)
    genre = np.asarray(genre)

    data = ((mel, perc, harm), genre)
    return data

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
            cur = ' | {} | {} [{}/{}]'.format(self.i, self.song, self.num,
                                            self.n)
            return cur
        else:
            raise StopIteration()

def train_model(x_np_array, y_genre):
    # LSTM for sequence classification in the IMDB dataset
    import numpy as np
    import keras

    from keras.datasets import imdb
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence
    # fix random seed for reproducibility
    np.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000

    from sklearn.model_selection import train_test_split
    # X, y = np.arrange(10).reshape((5, 2)), range(5)

    X_train, x_test, y_train, y_test = \
        train_test_split(x_np_array, y_genre, test_size = .33)

    layer_sizes = [300,144,36]

    # Keras uses the Sequential model for linear stacking of layers.
    # That is, creating a neural network is as easy as (later) defining the layers!
    from keras.models import Sequential
    model = Sequential()

    # Use the dropout regularization method
    from keras.layers import Dropout

    # Now that we have the model, let's add some layers:
    from keras.layers.core import Dense, Activation
    # Everything we've talked about in class so far is referred to in
    # Keras as a "dense" connection between layers, where every input
    # unit connects to a unit in the next layer

    # First a fully-connected (Dense) hidden layer with appropriate input
    # dimension, 577 INPUTS AND 300 OUTPUTS, and ReLU activation
    # THIS IS THE INPUT LAYER
    model.add(Dense(
        input_dim=X_train.shape[1], output_dim=layer_sizes[0]
    ))
    model.add(Activation('relu'))

    # ADD DROPOUT --> MUST DECIDE PERCENTAGE OF INPUT UNITS TO DROPOUT
    model.add(Dropout(.2))

    # Now our second hidden layer with 300 inputs (from the first
    # hidden layer) and 144 outputs. Also with ReLU activation
    # THIS IS HIDDEN LAYER
    model.add(Dense(
        input_dim=layer_sizes[0], output_dim=layer_sizes[1]
    ))
    model.add(Activation('relu'))

    # ADD DROPOUT
    model.add(Dropout(.2))

    # THIRD HIDDEN LAYER WITH 144 INPUTS AND 36 OUTPUTS, RELU ACTIVATION
    # Also with ReLU activation
    # THIS IS HIDDEN LAYER
    model.add(Dense(
        input_dim=layer_sizes[1], output_dim=layer_sizes[2]
    ))
    model.add(Activation('relu'))

    # ADD DROPOUT
    model.add(Dropout(.2))

    # Finally, add a readout layer, mapping the 5 hidden units
    # to two output units using the softmax function
    # THIS IS OUR OUTPUT LAYER
    model.add(Dense(output_dim=np.unique(y_train).shape[0], init='uniform'))
    model.add(Activation('softmax'))

    # Next we let the network know how to learn
    from keras.optimizers import SGD
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    # Before we can fit the network, we have to one-hot vectorize our response.
    # Fortunately, there is a keras method for that.
    from keras.utils.np_utils import to_categorical
    # for each of our 8 categories, map an output
    # Must first convert each category string to consistent ints
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(y_train)
    np.save('classes.npy', encoder.classes_)
    encoded_y_train = encoder.transform(y_train)
    encoded_y_test = encoder.transform(y_test)
    y_train_vectorized = to_categorical(encoded_y_train)

    # print out shape
    # y_train_vectorized.shape

    # remember that the bigger the nb_epoch the better the fit (so go bigger than 50)
    model.fit(X_train, y_train_vectorized, nb_epoch=1000, batch_size=20,
              verbose=0)

    # now our neural network works like a scikit-learn classifier
    proba = model.predict_proba(X_test, batch_size=32)

    # Print the accuracy:
    from sklearn.metrics import accuracy_score
    classes = np.argmax(proba, axis=1)
    print("The neural network model has an accuracy score of:",
          accuracy_score(encoded_y_test, classes))

    return model


if __name__ == '__main__':
    main()