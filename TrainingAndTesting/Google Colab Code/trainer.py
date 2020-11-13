import torch
import time
import pickle
import numpy as np


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

# Turn data into a 1d array, padd
oneDInputData = list()
for i in range(len(inputData)):
    inputData[i] = np.pad(inputData[i], ((0, 0), (maxDur - inputData[i].shape[1], 0)), 'constant')
    oneDInputData.append(inputData[i].flatten())
inputData = np.array(oneDInputData)

print("inputData converted")

# Initialize model
model = torch.nn.Sequential(
    torch.nn.Linear(51600, 500),
    torch.nn.Sigmoid(),
    torch.nn.Linear(500, 30),
    torch.nn.Sigmoid(),
    torch.nn.Linear(30, 2),
    torch.nn.Sigmoid()
)

lossFunction = torch.nn.SmoothL1Loss()
epochs = 10
learningRate = .01
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)




# Create a list of batches of 10
arrayOfInputs = list(chunks(inputData, 10))
print(len(arrayOfInputs))
arrayOfOutputs = list(chunks(expectedOutputs, 10))

# Loop through batches
for i in range(len(arrayOfInputs)):
    # print(np.array(arrayOfInputs)[i])
    inputData = torch.tensor(arrayOfInputs[i]).float()
    expectedOutputs = torch.tensor(arrayOfOutputs[i]).float()

    print("batch " + str(i))

    # Loop through epochs
    for t in range(epochs):
        predictedOutput = model(inputData)

        # Calculate loss
        loss = lossFunction(predictedOutput, expectedOutputs)

        # Print t and loss for debugging purposes
        if t % 10 == 0:
            print("t= " + str(t) + " batch = " + str(i) + "/" + str(len(arrayOfInputs)) + " loss = " + str(loss.item()))

        # Zero gradients before backwards pass
        optimizer.zero_grad()

        # Compute gradient of loss with respect to parameters
        loss.backward()

        # Update weights
        optimizer.step()

        # After each epoch, save the model in case of crash or disconnect from Colab server
        pickle.dump(model, open("/content/gdrive/My Drive/NN Models/model.txt", "wb"))

end = time.time()

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