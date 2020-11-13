import pathlib

import librosa
import torch
import pandas as pd
import matplotlib as plt
import time
import pickle
import numpy as np


# torch.cuda.empty_cache()

# from AudioClassifier import AudioClassifier
# from DataProcessor import DataPreprocessor

# First we need to iterate through all our dataset folders. For each of the folders (let's call them F1), there is a CSV and subfolders (calle them F2).
# Load some audio file (maybe start with the deep cough audio files first) into memory by appending its mfcc values as a single object to a list and
# appending a set of values {x,y} to another list depending on whether the csv files say the person is positive or negative for covid. If the person is currently positive,
# {x,y} = {1, 0}. If they're negative {x, y} = {0, 1}. For each F1, we train our model, then we clear both lists and move to the next F1 until done.
# Once finished, we save our model and test. We'll save folder 20200525 as our testing data set


# We repeat this for each feature that we're looking at: phonation, breathing, deep cough, shallow cough, etc

# If the audio feature is missing, then don't append anything to either list

# First we'll start by only training one model. Once we get that working, we'll train a few models that look at different features, then implement stacking by
# creating a metalearner that we train over our set of audio-classifiers.

# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


start = time.time()

inputData = pickle.load(open("/content/gdrive/My Drive/NN Models/coswaraInputs.txt", "rb"))
expectedOutputs = pickle.load(open("/content/gdrive/My Drive/NN Models/coswaraExpectedOutputs.txt", "rb"))

print(len(inputData))

maxDur = 0
minDur = 999999
for datum in inputData:
    maxDur = max(datum.shape[1], maxDur)
    minDur = min(minDur, datum.shape[1])
    # print(datum.shape[1])
print("max = " + str(maxDur))
print("min = " + str(minDur))

oneDInputData = list()
for i in range(len(inputData)):
    inputData[i] = np.pad(inputData[i], ((0, 0), (maxDur - inputData[i].shape[1], 0)), 'constant')
    # print(inputData[i].shape)
    oneDInputData.append(inputData[i].flatten())

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ac = AudioClassifier(20, .01)


model = torch.nn.Sequential(
    torch.nn.Linear(51600, 500),
    torch.nn.Sigmoid(),
    torch.nn.Linear(500, 30),
    torch.nn.Sigmoid(),
    torch.nn.Linear(30, 2),
    torch.nn.Sigmoid()
)

# lossFunction = torch.nn.MSELoss(reduction='sum')
lossFunction = torch.nn.SmoothL1Loss()
epochs = 10
learningRate = .01
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

# model.cuda()

inputData = np.array(oneDInputData)
# inputData = torch.tensor(inputData).float().cuda()
# expectedOutputs = torch.tensor(expectedOutputs).float().cuda()
print("inputData converted")

arrayOfInputs = list(chunks(inputData, 10))
print(len(arrayOfInputs))
arrayOfOutputs = list(chunks(expectedOutputs, 10))

for i in range(len(arrayOfInputs)):
    # print(np.array(arrayOfInputs)[i])
    inputData = torch.tensor(arrayOfInputs[i]).float()
    expectedOutputs = torch.tensor(arrayOfOutputs[i]).float()

    print("batch " + str(i))

    for t in range(epochs):

        # print("Epoch: " +str(1))

        # Predict output from inputdata
        predictedOutput = model(inputData)

        # print("predicted")

        # Calculate loss
        loss = lossFunction(predictedOutput, expectedOutputs)

        # print("calculated loss)")

        # Print t and loss for debugging purposes
        if t % 10 == 0:
            print("t= " + str(t) + " batch = " + str(i) + "/" + str(len(arrayOfInputs)) + " loss = " + str(loss.item()))

        # Zero gradients before backwards pass
        optimizer.zero_grad()

        # print("zeroed grad")

        # Compute gradient of loss with respect to parameters
        loss.backward()

        # print("back pass")

        # Update weights
        optimizer.step()

        pickle.dump(model, open("/content/gdrive/My Drive/NN Models/model.txt", "wb"))

# ac.train(torch.tensor(inputData), torch.tensor(expectedOutputs), True)


end = time.time()

# inputData = dp.getInputData()
# expectedOutputs = dp.getExpectedOutputs()

# pickle.dump(inputData, open("coswaraInputs.txt", "wb"))
# pickle.dump(expectedOutputs, open("coswaraExpectedOutputs.txt", "wb"))

print("This whole test took " + str(end - start) + " seconds")
exit(0)

# Notebook: https://colab.research.google.com/drive/1y6kcWj_XJTO8wTQlqPWhXvXapTsH9WAL?usp=sharing

# Mount g drive:
# from google.colab import drive
# drive.mount('/content/gdrive')

# Notebook: https://colab.research.google.com/drive/1y6kcWj_XJTO8wTQlqPWhXvXapTsH9WAL?usp=sharing

# Mount g drive:
# from google.colab import drive
# drive.mount('/content/gdrive')