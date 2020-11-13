import torch
import time
import pickle
import numpy as np

# File for testing

start = time.time()

# x = True
# dp = DataPreprocessor('cough-heavy.wav', True)
# while x == True:
    # x = dp.getF1Data()

inputData = pickle.load(open("../SerializedData/coswaraInputs.txt", "rb"))
expectedOutputs = pickle.load(open("../SerializedData/coswaraExpectedOutputs.txt", "rb"))

maxDur = 0
minDur = 999999
for datum in inputData:
    maxDur = max(datum.shape[1], maxDur)
    minDur = min(minDur,datum.shape[1])
    print(datum.shape[1])
print("max = " +str(maxDur))
print("min = " +str(minDur))

oneDInputData = list()
for i in range(len(inputData)):
    inputData[i] = np.pad(inputData[i], ((0, 0), (maxDur - inputData[i].shape[1], 0)), 'constant')
    print(inputData[i].shape)
    oneDInputData.append(inputData[i].flatten())


inputData = np.array(oneDInputData)
inputData = torch.tensor(inputData).float()
expectedOutputs = torch.tensor(expectedOutputs).float()
print("inputData converted")


model = torch.nn.Sequential(
            torch.nn.Linear(51600, 30000),
            torch.nn.Sigmoid(),
            torch.nn.Linear(30000, 500),
            torch.nn.Sigmoid(),
            torch.nn.Linear(500, 2),
            torch.nn.Sigmoid()
)

lossFunction = torch.nn.MSELoss(reduction='sum')
epochs = 1
learningRate = .01
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

for t in range(epochs):

    print("Epoch: " +str(1))

    # Predict output from inputdata
    predictedOutput = model(inputData)

    print("predicted")

    # Calculate loss
    loss = lossFunction(predictedOutput, expectedOutputs)

    print("calculated loss)")

    # Print t and loss for debugging purposes
    if t % 100 == 0:
        print(t, loss.item())

    # Zero gradients before backwards pass
    optimizer.zero_grad()

    print("zeroed grad")

    # Compute gradient of loss with respect to parameters
    loss.backward()

    print("back pass")

    # Update weights
    optimizer.step()


end = time.time()

# inputData = dp.getInputData()
# expectedOutputs = dp.getExpectedOutputs()

# pickle.dump(inputData, open("coswaraInputs.txt", "wb"))
# pickle.dump(expectedOutputs, open("coswaraExpectedOutputs.txt", "wb"))

print("This whole test took " + str(end - start) + " seconds")
exit(0)
