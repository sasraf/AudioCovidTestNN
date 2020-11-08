import torch

class AudioClassifier:
    def __init__(self, inputEpochs, learningRate):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(40, 15),
            torch.nn.ReLU(),
            torch.nn.Linear(15,2),
            torch.nn.Softmax()
        )

        self.lossFunction = torch.nn.MSELoss()
        self.epochs = inputEpochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)

    def feedForward(self, inputArray):
        return self.model(inputArray)

    def train(self, inputData, expectedOutputs, debug):
        for t in range(self.epochs):

            # Predict output from inputdata
            predictedOutput = self.model(inputData)

            # Calculate loss
            self.loss = self.lossFunction(predictedOutput, expectedOutputs)

            # Print t and loss for debugging purposes
            if debug and t % 100 == 0:
                print(t, self.loss.item())

            # Zero gradients before backwards pass
            self.optimizer.zero_grad()

            # Compute gradient of loss with respect to parameters
            self.loss.backward()

            # Update weights
            self.optimizer.step()



