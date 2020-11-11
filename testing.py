import librosa
import torch
import pandas as pd
import matplotlib as plt
import time
import pickle

from DataProcessor import DataPreprocessor

# First we need to iterate through all our dataset folders. For each of the folders (let's call them F1), there is a CSV and subfolders (calle them F2).
# Load some audio file (maybe start with the deep cough audio files first) into memory by appending its mfcc values as a single object to a list and
# appending a set of values {x,y} to another list depending on whether the csv files say the person is positive or negative for covid. If the person is currently positive,
# {x,y} = {1, 0}. If they're negative {x, y} = {0, 1}. For each F1, we train our model, then we clear both lists and move to the next F1 until done.
# Once finished, we save our model and test. We'll save folder 20200525 as our testing data set


# We repeat this for each feature that we're looking at: phonation, breathing, deep cough, shallow cough, etc

# If the audio feature is missing, then don't append anything to either list

# First we'll start by only training one model. Once we get that working, we'll train a few models that look at different features, then implement stacking by
# creating a metalearner that we train over our set of audio-classifiers.


start = time.time()

x = True

dp = DataPreprocessor('cough-heavy.wav', True)

while x == True:
    x = dp.getF1Data()

# TODO: pickle the processed data
end = time.time()

print(x)

print("This whole test took " + str(end - start) + " seconds")



# Librosa preprocessing example

# fn = 'data/20200413/aGEXEhp3mbUandZBtCuEooDQrK53/cough-heavvy.wav'
#
# librosa_audio, librosa_sample_rate = librosa.load(fn)
# mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc = 40)
#
#
# print(mfccs.shape)
# print(mfccs)
#
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfccs, x_axis='time')
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()
#

exit(0)
