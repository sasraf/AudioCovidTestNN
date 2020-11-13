import torch
import pickle
import numpy as np

# Load model, data
model = pickle.load(open("../Models/model0sPaddedAtEnd.txt", "rb"))
inputData = pickle.load(open("../SerializedData/coswaraInputs.txt", "rb"))
expectedOutputs = pickle.load(open("../SerializedData/coswaraExpectedOutputs.txt", "rb"))

# Determine longest file in our data
maxDur = 0
minDur = 999999
for datum in inputData:
    maxDur = max(datum.shape[1], maxDur)
    minDur = min(minDur, datum.shape[1])
    print(datum.shape[1])
print("max = " + str(maxDur))
print("min = " + str(minDur))

# Pad all our data
oneDInputData = list()
for i in range(len(inputData)):
    inputData[i] = np.pad(inputData[i], ((0, 0), (maxDur - inputData[i].shape[1], 0)), 'constant')
    oneDInputData.append(inputData[i].flatten())

# Predict outputs with our model
predictions = model(torch.tensor(torch.tensor(np.array(oneDInputData)).float()))

# Calculate accuracy
accuracy = 0
for i in range(len(predictions)):
    actual = expectedOutputs[i]
    curPrediction = predictions[i]
    # print(curPrediction)
    positive = curPrediction.data[0]
    negative = curPrediction.data[1]
    actualPositive = actual[0]
    actualNegative = actual[1]

    if positive >= negative:
        if actualPositive > actualNegative:
            accuracy += 1
    elif negative > positive:
        if actualNegative > actualPositive:
            accuracy += 1
print("Accuracy: ", accuracy/len(predictions))
