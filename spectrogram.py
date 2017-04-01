from __future__ import print_function

import numpy as np 
import scipy as sp 
# import keras only if we need DNN
import matplotlib.pyplot as plt

#import IPython.display

# Librosa is a music analysis tool
# https://github.com/librosa/librosa
import librosa as lb
import librosa.display

y, sr = lb.load(path='test.mp3')

# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = lb.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power as reference.
log_S = lb.logamplitude(S, ref_power=np.max)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
lb.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()

# display
plt.show()